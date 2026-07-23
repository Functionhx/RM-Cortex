"""Event-derived dense rewards; referee state never depends on rewards."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.schema import Winner, unit_teams
from rm_referee.state import GameState


@dataclass(frozen=True)
class RewardConfig:
    damage_scale: float = 0.01
    kill_reward: float = 2.0
    death_penalty: float = 1.0
    technology_reward: float = 0.5
    rune_reward: float = 0.75
    dart_hit_reward: float = 0.5
    terminal_reward: float = 10.0
    invalid_overfire_penalty: float = 0.05


class RewardBuilder:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def build(self, game: GameState, events: RefereeEvents) -> Tensor:
        reward = events.damage_dealt * self.config.damage_scale
        reward -= events.deaths.to(game.dtype) * self.config.death_penalty
        reward -= events.overfire.sum(dim=-1).to(game.dtype) * self.config.invalid_overfire_penalty

        valid_kill = events.kill_source >= 0
        kill_source = torch.clamp(events.kill_source, min=0)
        kill_reward = torch.zeros_like(reward)
        kill_reward.scatter_add_(
            1,
            kill_source,
            valid_kill.to(game.dtype) * self.config.kill_reward,
        )
        reward += kill_reward

        team_event = (
            (events.tech_completed > 0).to(game.dtype) * self.config.technology_reward
            + events.rune_activated.to(game.dtype) * self.config.rune_reward
            + events.dart_hits.to(game.dtype) * self.config.dart_hit_reward
        )
        reward += team_event.repeat_interleave(constants.ROLES_PER_TEAM, dim=1)

        teams = unit_teams(game.device)
        own_winner = game.winner[:, None] == teams[None, :]
        decided = events.match_ended[:, None] & (game.winner[:, None] != Winner.DRAW)
        terminal = torch.where(
            own_winner,
            torch.full_like(reward, self.config.terminal_reward),
            torch.full_like(reward, -self.config.terminal_reward),
        )
        reward += torch.where(decided, terminal, torch.zeros_like(terminal))
        return reward
