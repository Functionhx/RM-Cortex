"""Stable integer schema shared by referee and environment backends."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor

from rm_referee.constants import ROLES_PER_TEAM, UNIT_COUNT


class Team(IntEnum):
    RED = 0
    BLUE = 1


class Role(IntEnum):
    HERO = 0
    ENGINEER = 1
    INFANTRY_3 = 2
    INFANTRY_4 = 3
    AERIAL = 4
    SENTRY = 5
    BASE = 6
    OUTPOST = 7


class Weapon(IntEnum):
    MM17 = 0
    MM42 = 1


class HeatLock(IntEnum):
    NONE = 0
    TEMPORARY = 1
    PERMANENT = 2


class Winner(IntEnum):
    UNDECIDED = -1
    RED = 0
    BLUE = 1
    DRAW = 2


class PurchaseKind(IntEnum):
    NONE = 0
    LOCAL = 1
    REMOTE = 2


class RadarQuality(IntEnum):
    WRONG = -1
    HALF_ACCURATE = 0
    ACCURATE = 1
    UNKNOWN = 2


def slot(team: int | Team, role: int | Role) -> int:
    """Return the absolute unit slot for a team/role pair."""

    return int(team) * ROLES_PER_TEAM + int(role)


def unit_teams(device: torch.device | str | None = None) -> Tensor:
    return torch.arange(UNIT_COUNT, device=device, dtype=torch.long) // ROLES_PER_TEAM


def unit_roles(device: torch.device | str | None = None) -> Tensor:
    return torch.arange(UNIT_COUNT, device=device, dtype=torch.long) % ROLES_PER_TEAM


def ground_robot_mask(device: torch.device | str | None = None) -> Tensor:
    roles = unit_roles(device)
    return (
        (roles == Role.HERO)
        | (roles == Role.ENGINEER)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.SENTRY)
    )


def weapon_capability(device: torch.device | str | None = None) -> Tensor:
    """Return ``[unit, weapon]`` launcher availability."""

    roles = unit_roles(device)
    result = torch.zeros((UNIT_COUNT, 2), device=device, dtype=torch.bool)
    result[:, Weapon.MM17] = (
        (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.AERIAL)
        | (roles == Role.SENTRY)
    )
    result[:, Weapon.MM42] = roles == Role.HERO
    return result


def chassis_power_mask(device: torch.device | str | None = None) -> Tensor:
    roles = unit_roles(device)
    return (
        (roles == Role.HERO)
        | (roles == Role.ENGINEER)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.SENTRY)
    )


def chassis_energy_mask(device: torch.device | str | None = None) -> Tensor:
    roles = unit_roles(device)
    return (
        (roles == Role.HERO)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.SENTRY)
    )
