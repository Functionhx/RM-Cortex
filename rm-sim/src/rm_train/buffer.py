"""On-policy rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rm_referee import constants
from rm_train.policy import PolicyAction


@dataclass
class FlatRolloutBatch:
    observations: Tensor
    central_state: Tensor
    target_mask: Tensor
    fire_mask: Tensor
    agent_ids: Tensor
    actions: PolicyAction
    old_log_prob: Tensor
    old_value: Tensor
    returns: Tensor
    advantages: Tensor

    @property
    def size(self) -> int:
        return self.old_log_prob.shape[0]

    def take(self, indices: Tensor) -> "FlatRolloutBatch":
        return FlatRolloutBatch(
            observations=self.observations[indices],
            central_state=self.central_state[indices],
            target_mask=self.target_mask[indices],
            fire_mask=self.fire_mask[indices],
            agent_ids=self.agent_ids[indices],
            actions=self.actions.take(indices),
            old_log_prob=self.old_log_prob[indices],
            old_value=self.old_value[indices],
            returns=self.returns[indices],
            advantages=self.advantages[indices],
        )


@dataclass
class RolloutBatch:
    observations: Tensor
    central_state: Tensor
    target_mask: Tensor
    fire_mask: Tensor
    actions: PolicyAction
    old_log_prob: Tensor
    old_value: Tensor
    returns: Tensor
    advantages: Tensor

    def flatten(self) -> FlatRolloutBatch:
        time_steps, num_envs, num_agents = self.old_log_prob.shape
        central = self.central_state[:, :, None, :].expand(
            time_steps,
            num_envs,
            num_agents,
            -1,
        )
        agent_ids = (
            torch.arange(num_agents, device=self.old_log_prob.device)
            .view(1, 1, num_agents)
            .expand(time_steps, num_envs, -1)
        )
        return FlatRolloutBatch(
            observations=self.observations.reshape(-1, self.observations.shape[-1]),
            central_state=central.reshape(-1, central.shape[-1]),
            target_mask=self.target_mask.reshape(-1, self.target_mask.shape[-1]),
            fire_mask=self.fire_mask.reshape(-1),
            agent_ids=agent_ids.reshape(-1),
            actions=self.actions.flatten(),
            old_log_prob=self.old_log_prob.reshape(-1),
            old_value=self.old_value.reshape(-1),
            returns=self.returns.reshape(-1),
            advantages=self.advantages.reshape(-1),
        )


@dataclass
class RolloutStorage:
    observations: list[Tensor] = field(default_factory=list)
    central_state: list[Tensor] = field(default_factory=list)
    target_mask: list[Tensor] = field(default_factory=list)
    fire_mask: list[Tensor] = field(default_factory=list)
    actions: list[PolicyAction] = field(default_factory=list)
    log_prob: list[Tensor] = field(default_factory=list)
    value: list[Tensor] = field(default_factory=list)
    reward: list[Tensor] = field(default_factory=list)
    done: list[Tensor] = field(default_factory=list)

    def add(
        self,
        *,
        observations: Tensor,
        central_state: Tensor,
        target_mask: Tensor,
        fire_mask: Tensor,
        actions: PolicyAction,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> None:
        self.observations.append(observations.detach())
        self.central_state.append(central_state.detach())
        self.target_mask.append(target_mask.detach())
        self.fire_mask.append(fire_mask.detach())
        self.actions.append(
            PolicyAction(
                **{name: getattr(actions, name).detach() for name in actions.__dataclass_fields__}
            )
        )
        self.log_prob.append(log_prob.detach())
        self.value.append(value.detach())
        self.reward.append(reward.detach())
        self.done.append(done.detach())

    def finish(
        self,
        next_value: Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        if not self.reward:
            raise ValueError("cannot finish an empty rollout")
        rewards = torch.stack(self.reward)
        values = torch.stack(self.value)
        dones = torch.stack(self.done).to(values.dtype)
        if dones.ndim == 2:
            dones = dones[:, :, None].expand_as(values)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)
        for time_index in range(rewards.shape[0] - 1, -1, -1):
            future_value = (
                next_value if time_index == rewards.shape[0] - 1 else values[time_index + 1]
            )
            nonterminal = 1.0 - dones[time_index]
            delta = rewards[time_index] + gamma * future_value * nonterminal - values[time_index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[time_index] = gae
        returns = advantages + values
        if rewards.shape[-1] != constants.UNIT_COUNT:
            raise ValueError("rollout reward must use the 16-agent layout")
        return RolloutBatch(
            observations=torch.stack(self.observations),
            central_state=torch.stack(self.central_state),
            target_mask=torch.stack(self.target_mask),
            fire_mask=torch.stack(self.fire_mask),
            actions=PolicyAction.stack(self.actions),
            old_log_prob=torch.stack(self.log_prob),
            old_value=values,
            returns=returns,
            advantages=advantages,
        )
