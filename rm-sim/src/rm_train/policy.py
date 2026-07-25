"""Parameter-shared heterogeneous actor and centralized critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical, Normal

from rm_referee import constants
from rm_referee.schema import Role
from rm_world.geometry import NO_TARGET
from rm_world.observations import ENTITY_DIM


@dataclass
class PolicyAction:
    """Raw policy samples before deterministic world-action decoding."""

    motion: Tensor
    target: Tensor
    fire: Tensor
    common: Tensor
    special: Tensor
    mode: Tensor
    radar_target: Tensor
    radar_report: Tensor

    @classmethod
    def stack(cls, actions: list["PolicyAction"]) -> "PolicyAction":
        if not actions:
            raise ValueError("cannot stack an empty action list")
        return cls(
            **{
                name: torch.stack([getattr(action, name) for action in actions])
                for name in cls.__dataclass_fields__
            }
        )

    def flatten(self) -> "PolicyAction":
        """Flatten rollout, environment, and agent axes into one batch."""

        scalar_fields = {"target", "fire", "mode", "radar_target"}
        return PolicyAction(
            **{
                name: (
                    value.reshape(-1)
                    if name in scalar_fields
                    else value.reshape(-1, value.shape[-1])
                )
                for name, value in (
                    (field, getattr(self, field)) for field in self.__dataclass_fields__
                )
            }
        )

    def take(self, indices: Tensor) -> "PolicyAction":
        return PolicyAction(
            **{name: getattr(self, name)[indices] for name in self.__dataclass_fields__}
        )


@dataclass
class PolicyStep:
    action: PolicyAction
    log_prob: Tensor
    entropy: Tensor
    value: Tensor


@dataclass
class PolicyDistributions:
    motion: Normal
    target: Categorical
    fire: Bernoulli
    common: Bernoulli
    special: Bernoulli
    mode: Categorical
    radar_target: Categorical
    radar_report: Normal


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class SharedMAPPOPolicy(nn.Module):
    """One actor for all roles and a privileged per-agent value function."""

    CHECKPOINT_SCHEMA_VERSION = 3

    def __init__(
        self,
        *,
        observation_dim: int = 34,
        entity_dim: int = ENTITY_DIM,
        state_dim: int = 117,
        hidden_dim: int = 128,
        role_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.entity_dim = entity_dim
        self.state_dim = state_dim
        self.role_embedding = nn.Embedding(constants.ROLES_PER_TEAM, role_embedding_dim)
        self.team_embedding = nn.Embedding(constants.TEAM_COUNT, role_embedding_dim // 2)
        identity_dim = role_embedding_dim + role_embedding_dim // 2

        self.entity_encoder = _mlp(
            entity_dim + 1,
            hidden_dim,
            hidden_dim,
        )
        self.entity_query = nn.Linear(observation_dim + identity_dim, hidden_dim)
        self.entity_key = nn.Linear(hidden_dim, hidden_dim)
        self.entity_value = nn.Linear(hidden_dim, hidden_dim)
        self.actor_body = _mlp(
            observation_dim + identity_dim + hidden_dim,
            hidden_dim,
            hidden_dim,
        )
        self.motion_mean = nn.Linear(hidden_dim, 4)
        self.motion_log_std = nn.Parameter(torch.full((4,), -0.5))
        self.target_logits = nn.Linear(hidden_dim, 9)
        self.fire_logits = nn.Linear(hidden_dim, 1)
        self.common_logits = nn.Linear(hidden_dim, 2)
        self.special_logits = nn.Linear(hidden_dim, 3)
        self.mode_logits = nn.Linear(hidden_dim, 5)
        # The final class is an explicit no-report action. A radar station must
        # be able to withhold a low-confidence coordinate instead of being
        # forced to submit one every policy transition.
        self.radar_target_logits = nn.Linear(hidden_dim, constants.UNIT_COUNT + 1)
        self.radar_report_mean = nn.Linear(hidden_dim, 2)
        self.radar_report_log_std = nn.Parameter(torch.full((2,), -0.5))

        self.critic = _mlp(
            state_dim + identity_dim,
            hidden_dim,
            1,
        )

    def architecture_kwargs(self) -> dict[str, int]:
        """Return the constructor arguments required to reload this policy."""

        return {
            "observation_dim": self.observation_dim,
            "entity_dim": self.entity_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.entity_query.out_features,
            "role_embedding_dim": self.role_embedding.embedding_dim,
        }

    @staticmethod
    def default_agent_ids(
        batch_shape: tuple[int, ...],
        device: torch.device,
    ) -> Tensor:
        if not batch_shape or batch_shape[-1] != constants.UNIT_COUNT:
            raise ValueError("the final batch axis must contain the 16 agent slots")
        view_shape = (1,) * (len(batch_shape) - 1) + (constants.UNIT_COUNT,)
        return (
            torch.arange(constants.UNIT_COUNT, device=device).view(view_shape).expand(batch_shape)
        )

    def _identity(self, agent_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        roles = agent_ids.remainder(constants.ROLES_PER_TEAM)
        teams = torch.div(agent_ids, constants.ROLES_PER_TEAM, rounding_mode="floor")
        identity = torch.cat(
            (self.role_embedding(roles), self.team_embedding(teams)),
            dim=-1,
        )
        return identity, roles, teams

    def entity_attention(
        self,
        observations: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        agent_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode entity beliefs while retaining current visibility as a feature."""

        if observations.shape[-1] != self.observation_dim:
            raise ValueError("actor observation dimension does not match the policy")
        if entities.ndim != observations.ndim + 1:
            raise ValueError("entities need one target axis beyond the actor observations")
        if entities.shape[:-2] != observations.shape[:-1]:
            raise ValueError("entity batch axes must match the actor observations")
        if entities.shape[-2] != constants.UNIT_COUNT:
            raise ValueError("the entity target axis must contain the 16 unit slots")
        if entities.shape[-1] != self.entity_dim:
            raise ValueError("entity feature dimension does not match the policy")
        if entity_mask.shape != entities.shape[:-1]:
            raise ValueError("entity mask must match the entity batch and target axes")
        if entity_mask.dtype is not torch.bool:
            raise ValueError("entity mask must use bool dtype")
        if agent_ids is None:
            agent_ids = self.default_agent_ids(observations.shape[:-1], observations.device)
        if agent_ids.shape != observations.shape[:-1]:
            raise ValueError("agent ids must match the actor observation batch axes")

        identity, _, _ = self._identity(agent_ids)
        actor_input = torch.cat((observations, identity), dim=-1)
        source_feature = entity_mask.to(entities.dtype).unsqueeze(-1)
        encoded = self.entity_encoder(torch.cat((entities, source_feature), dim=-1))
        query = self.entity_query(actor_input).unsqueeze(-2)
        keys = self.entity_key(encoded)
        values = self.entity_value(encoded)
        scale = float(keys.shape[-1]) ** -0.5
        weights = torch.softmax((query * keys).sum(dim=-1) * scale, dim=-1)
        context = (weights.unsqueeze(-1) * values).sum(dim=-2)
        return context, weights

    def _actor_outputs(
        self,
        observations: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        agent_ids: Tensor,
    ) -> tuple[Tensor, ...]:
        features, roles, _ = self.actor_features(
            observations,
            entities,
            entity_mask,
            agent_ids,
        )
        return (
            features,
            roles,
            self.motion_mean(features),
            self.target_logits(features),
            self.fire_logits(features).squeeze(-1),
            self.common_logits(features),
            self.special_logits(features),
            self.mode_logits(features),
            self.radar_target_logits(features),
            self.radar_report_mean(features),
        )

    def actor_features(
        self,
        observations: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        agent_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return shared actor features, role indices, and entity attention.

        This is the stable transfer boundary for auxiliary pretraining. Action
        heads and the centralized critic intentionally remain outside it.
        """

        identity, roles, _ = self._identity(agent_ids)
        entity_context, attention = self.entity_attention(
            observations,
            entities,
            entity_mask,
            agent_ids,
        )
        features = self.actor_body(torch.cat((observations, identity, entity_context), dim=-1))
        return features, roles, attention

    def value(
        self,
        central_state: Tensor,
        agent_ids: Tensor,
    ) -> Tensor:
        if central_state.shape[-1] != self.state_dim:
            raise ValueError("central state dimension does not match the policy")
        if central_state.shape[:-1] != agent_ids.shape:
            if central_state.shape[:-1] != agent_ids.shape[:-1]:
                raise ValueError("central state cannot be broadcast across agents")
            central_state = central_state.unsqueeze(-2).expand(*agent_ids.shape, self.state_dim)
        identity, _, _ = self._identity(agent_ids)
        return self.critic(torch.cat((central_state, identity), dim=-1)).squeeze(-1)

    @staticmethod
    def _head_masks(roles: Tensor, fire_mask: Tensor) -> tuple[Tensor, ...]:
        robot = roles <= Role.SENTRY
        ground = robot & (roles != Role.AERIAL)
        launcher = (
            (roles == Role.HERO)
            | (roles == Role.INFANTRY_3)
            | (roles == Role.INFANTRY_4)
            | (roles == Role.AERIAL)
            | (roles == Role.SENTRY)
        )
        combat = launcher & fire_mask
        special = (
            (roles == Role.HERO)
            | (roles == Role.ENGINEER)
            | (roles == Role.AERIAL)
            | (roles == Role.SENTRY)
            | (roles == Role.BASE)
            | (roles == Role.OUTPOST)
        )
        mode = (roles == Role.ENGINEER) | (roles == Role.SENTRY) | (roles == Role.BASE)
        radar = roles == Role.OUTPOST
        return robot, ground, combat, special, mode, radar

    def _distributions(
        self,
        observations: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        agent_ids: Tensor,
        target_mask: Tensor,
    ) -> tuple[PolicyDistributions, Tensor]:
        (
            _,
            roles,
            motion_mean,
            target_logits,
            fire_logits,
            common_logits,
            special_logits,
            mode_logits,
            radar_target_logits,
            radar_report_mean,
        ) = self._actor_outputs(observations, entities, entity_mask, agent_ids)
        if target_mask.shape != (*agent_ids.shape, 9):
            raise ValueError("target mask must end in the nine-class target schema")
        if torch.any(~target_mask.any(dim=-1)):
            raise ValueError("every agent needs at least one valid target")
        masked_target_logits = target_logits.masked_fill(~target_mask, -1.0e9)
        motion_std = self.motion_log_std.clamp(-5.0, 1.0).exp().expand_as(motion_mean)
        radar_std = self.radar_report_log_std.clamp(-5.0, 1.0).exp().expand_as(radar_report_mean)
        distributions = PolicyDistributions(
            motion=Normal(motion_mean, motion_std),
            target=Categorical(logits=masked_target_logits),
            fire=Bernoulli(logits=fire_logits),
            common=Bernoulli(logits=common_logits),
            special=Bernoulli(logits=special_logits),
            mode=Categorical(logits=mode_logits),
            radar_target=Categorical(logits=radar_target_logits),
            radar_report=Normal(radar_report_mean, radar_std),
        )
        return distributions, roles

    def act(
        self,
        observations: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        central_state: Tensor,
        target_mask: Tensor,
        fire_mask: Tensor,
        *,
        deterministic: bool = False,
        agent_ids: Tensor | None = None,
    ) -> PolicyStep:
        if agent_ids is None:
            agent_ids = self.default_agent_ids(observations.shape[:-1], observations.device)
        distributions, roles = self._distributions(
            observations,
            entities,
            entity_mask,
            agent_ids,
            target_mask,
        )
        if deterministic:
            action = PolicyAction(
                motion=distributions.motion.mean,
                target=distributions.target.logits.argmax(dim=-1),
                fire=distributions.fire.logits >= 0,
                common=distributions.common.logits >= 0,
                special=distributions.special.logits >= 0,
                mode=distributions.mode.logits.argmax(dim=-1),
                radar_target=distributions.radar_target.logits.argmax(dim=-1),
                radar_report=distributions.radar_report.mean,
            )
        else:
            action = PolicyAction(
                motion=distributions.motion.sample(),
                target=distributions.target.sample(),
                fire=distributions.fire.sample().to(torch.bool),
                common=distributions.common.sample().to(torch.bool),
                special=distributions.special.sample().to(torch.bool),
                mode=distributions.mode.sample(),
                radar_target=distributions.radar_target.sample(),
                radar_report=distributions.radar_report.sample(),
            )
        _, _, combat, _, _, _ = self._head_masks(roles, fire_mask)
        action.target = torch.where(
            combat,
            action.target,
            torch.full_like(action.target, NO_TARGET),
        )
        action.fire &= combat
        log_prob, entropy = self._score(distributions, roles, fire_mask, action)
        return PolicyStep(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=self.value(central_state, agent_ids),
        )

    def evaluate_actions(
        self,
        observations: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        central_state: Tensor,
        target_mask: Tensor,
        fire_mask: Tensor,
        actions: PolicyAction,
        agent_ids: Tensor,
    ) -> PolicyStep:
        distributions, roles = self._distributions(
            observations,
            entities,
            entity_mask,
            agent_ids,
            target_mask,
        )
        log_prob, entropy = self._score(distributions, roles, fire_mask, actions)
        return PolicyStep(
            action=actions,
            log_prob=log_prob,
            entropy=entropy,
            value=self.value(central_state, agent_ids),
        )

    def _score(
        self,
        distributions: PolicyDistributions,
        roles: Tensor,
        fire_mask: Tensor,
        action: PolicyAction,
    ) -> tuple[Tensor, Tensor]:
        robot, ground, combat, special, mode, radar = self._head_masks(roles, fire_mask)

        motion_dimension = robot[..., None].expand_as(action.motion)
        common_dimension = ground[..., None].expand_as(action.common)
        special_dimension = special[..., None].expand_as(action.special)
        radar_report_active = radar & (action.radar_target < constants.UNIT_COUNT)
        radar_log_prob_dimension = radar_report_active[..., None].expand_as(action.radar_report)
        radar_report_probability = 1.0 - distributions.radar_target.probs[..., constants.UNIT_COUNT]
        radar_entropy_dimension = (radar.to(action.radar_report.dtype) * radar_report_probability)[
            ..., None
        ].expand_as(action.radar_report)

        log_prob = (
            (distributions.motion.log_prob(action.motion) * motion_dimension).sum(dim=-1)
            + distributions.target.log_prob(action.target) * combat
            + distributions.fire.log_prob(action.fire.to(torch.float32)) * combat
            + (
                distributions.common.log_prob(action.common.to(torch.float32)) * common_dimension
            ).sum(dim=-1)
            + (
                distributions.special.log_prob(action.special.to(torch.float32)) * special_dimension
            ).sum(dim=-1)
            + distributions.mode.log_prob(action.mode) * mode
            + distributions.radar_target.log_prob(action.radar_target) * radar
            + (
                distributions.radar_report.log_prob(action.radar_report) * radar_log_prob_dimension
            ).sum(dim=-1)
        )
        entropy = (
            (distributions.motion.entropy() * motion_dimension).sum(dim=-1)
            + distributions.target.entropy() * combat
            + distributions.fire.entropy() * combat
            + (distributions.common.entropy() * common_dimension).sum(dim=-1)
            + (distributions.special.entropy() * special_dimension).sum(dim=-1)
            + distributions.mode.entropy() * mode
            + distributions.radar_target.entropy() * radar
            + (distributions.radar_report.entropy() * radar_entropy_dimension).sum(dim=-1)
        )
        return log_prob, entropy
