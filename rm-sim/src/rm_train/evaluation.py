"""Deterministic policy evaluation against the scripted baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from rm_referee import constants
from rm_referee.schema import Team, Winner
from rm_train.actions import decode_policy_actions, replace_team_actions
from rm_train.policy import SharedMAPPOPolicy
from rm_world import ScriptedOpponent, TorchEnvConfig, TorchRMArena


@dataclass(frozen=True)
class EvaluationReport:
    completed_episodes: int
    red_wins: int
    blue_wins: int
    draws: int
    unfinished_environments: int
    mean_red_return: float
    mean_blue_return: float
    policy_steps: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@torch.no_grad()
def evaluate_against_scripted(
    policy: SharedMAPPOPolicy,
    *,
    num_envs: int = 8,
    max_policy_steps: int = 2100,
    device: str = "cpu",
    seed: int = 17,
) -> EvaluationReport:
    """Control red with MAPPO and blue with the deterministic opponent."""

    if num_envs <= 0 or max_policy_steps <= 0:
        raise ValueError("num_envs and max_policy_steps must be positive")
    environment = TorchRMArena(
        TorchEnvConfig(
            num_envs=num_envs,
            device=device,
            seed=seed,
            validate_referee=False,
        )
    )
    opponent = ScriptedOpponent()
    observation = environment.reset(seed=seed)
    policy = policy.to(device)
    policy.eval()
    team_return = torch.zeros(
        (num_envs, constants.TEAM_COUNT),
        device=environment.game.device,
    )
    finished = torch.zeros(num_envs, device=environment.game.device, dtype=torch.bool)
    winners = torch.full(
        (num_envs,),
        Winner.UNDECIDED,
        device=environment.game.device,
        dtype=torch.int8,
    )

    steps_taken = 0
    for steps_taken in range(1, max_policy_steps + 1):
        policy_step = policy.act(
            observation.agents,
            observation.entities,
            observation.entity_mask,
            observation.central,
            observation.target_mask,
            observation.fire_mask,
            deterministic=True,
        )
        actions = decode_policy_actions(
            environment.game,
            environment.world,
            policy_step.action,
        )
        scripted = opponent.act(
            environment.game,
            environment.world,
            team=Team.BLUE,
        )
        replace_team_actions(actions, scripted, Team.BLUE)
        result = environment.step(actions)
        per_team = result.reward.view(
            num_envs,
            constants.TEAM_COUNT,
            constants.ROLES_PER_TEAM,
        ).mean(dim=-1)
        team_return += per_team * (~finished)[:, None]
        newly_finished = result.terminated & ~finished
        winners[newly_finished] = environment.game.winner[newly_finished]
        finished |= result.terminated
        observation = result.observation
        if torch.all(finished):
            break

    completed = int(finished.sum().item())
    return EvaluationReport(
        completed_episodes=completed,
        red_wins=int(((winners == Winner.RED) & finished).sum().item()),
        blue_wins=int(((winners == Winner.BLUE) & finished).sum().item()),
        draws=int(((winners == Winner.DRAW) & finished).sum().item()),
        unfinished_environments=int((~finished).sum().item()),
        mean_red_return=float(team_return[:, Team.RED].mean().item()),
        mean_blue_return=float(team_return[:, Team.BLUE].mean().item()),
        policy_steps=steps_taken,
    )
