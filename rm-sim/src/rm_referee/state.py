"""Vectorized authoritative game state."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import (
    HeatLock,
    RadarQuality,
    Role,
    Winner,
    chassis_energy_mask,
    chassis_power_mask,
    ground_robot_mask,
    unit_roles,
)


@dataclass
class GameState:
    """Structure-of-arrays state with the environment dimension first."""

    hp: Tensor
    max_hp: Tensor
    alive: Tensor
    level: Tensor
    xp: Tensor

    heat: Tensor
    heat_limit: Tensor
    cooling_per_s: Tensor
    heat_lock: Tensor
    video_disabled: Tensor
    ammo: Tensor

    power_limit_w: Tensor
    power_buffer_j: Tensor
    chassis_disabled_s: Tensor
    chassis_energy: Tensor

    attack_multiplier: Tensor
    defense_fraction: Tensor
    vulnerability_fraction: Tensor
    radar_vulnerability_fraction: Tensor

    armor_last_hit_s: Tensor
    out_of_combat_s: Tensor
    respawn_progress: Tensor
    respawn_required: Tensor
    immediate_respawn_count: Tensor
    invulnerable_s: Tensor
    weak: Tensor

    team_coin: Tensor
    team_purchased_ammo: Tensor
    team_damage: Tensor
    base_shield: Tensor
    base_armor_deployed: Tensor
    outpost_destroyed_once: Tensor

    hero_42_silence_s: Tensor
    hero_42_invalid_block: Tensor

    radar_x: Tensor
    radar_p: Tensor
    radar_last_quality: Tensor

    elapsed_s: Tensor
    done: Tensor
    winner: Tensor

    @classmethod
    def create(
        cls,
        num_envs: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "GameState":
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        device = torch.device(device)
        env_unit = (num_envs, constants.UNIT_COUNT)
        env_team = (num_envs, constants.TEAM_COUNT)
        env_weapon = (num_envs, constants.UNIT_COUNT, constants.WEAPON_COUNT)

        role_hp = torch.tensor(
            constants.DEFAULT_MAX_HP * constants.TEAM_COUNT,
            device=device,
            dtype=dtype,
        )
        max_hp = role_hp.unsqueeze(0).expand(num_envs, -1).clone()
        hp = max_hp.clone()
        roles = unit_roles(device)
        alive = (max_hp > 0) & (roles.unsqueeze(0) != Role.AERIAL)

        default_heat_limit = torch.tensor(
            constants.DEFAULT_HEAT_LIMIT * constants.TEAM_COUNT,
            device=device,
            dtype=dtype,
        )
        default_cooling = torch.tensor(
            constants.DEFAULT_COOLING_PER_S * constants.TEAM_COUNT,
            device=device,
            dtype=dtype,
        )
        default_ammo = torch.tensor(
            constants.DEFAULT_AMMO * constants.TEAM_COUNT,
            device=device,
            dtype=torch.long,
        )
        default_power = torch.tensor(
            constants.DEFAULT_POWER_LIMIT_W * constants.TEAM_COUNT,
            device=device,
            dtype=dtype,
        )

        power_mask = chassis_power_mask(device).unsqueeze(0).expand(env_unit)
        energy_mask = chassis_energy_mask(device).unsqueeze(0).expand(env_unit)
        ground_mask = ground_robot_mask(device).unsqueeze(0).expand(env_unit)

        return cls(
            hp=hp,
            max_hp=max_hp,
            alive=alive,
            level=torch.ones(env_unit, device=device, dtype=torch.long),
            xp=torch.zeros(env_unit, device=device, dtype=dtype),
            heat=torch.zeros(env_weapon, device=device, dtype=dtype),
            heat_limit=default_heat_limit.unsqueeze(0).expand(env_weapon).clone(),
            cooling_per_s=default_cooling.unsqueeze(0).expand(env_weapon).clone(),
            heat_lock=torch.full(
                env_weapon,
                HeatLock.NONE,
                device=device,
                dtype=torch.int8,
            ),
            video_disabled=torch.zeros(env_unit, device=device, dtype=torch.bool),
            ammo=default_ammo.unsqueeze(0).expand(env_weapon).clone(),
            power_limit_w=default_power.unsqueeze(0).expand(env_unit).clone(),
            power_buffer_j=torch.where(
                power_mask,
                torch.full(env_unit, constants.POWER_BUFFER_MAX_J, device=device, dtype=dtype),
                torch.zeros(env_unit, device=device, dtype=dtype),
            ),
            chassis_disabled_s=torch.zeros(env_unit, device=device, dtype=dtype),
            chassis_energy=torch.where(
                energy_mask,
                torch.full(
                    env_unit,
                    constants.CHASSIS_ENERGY_INITIAL,
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros(env_unit, device=device, dtype=dtype),
            ),
            attack_multiplier=torch.ones(env_unit, device=device, dtype=dtype),
            defense_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            vulnerability_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            radar_vulnerability_fraction=torch.zeros(
                env_unit,
                device=device,
                dtype=dtype,
            ),
            armor_last_hit_s=torch.full(
                (
                    num_envs,
                    constants.UNIT_COUNT,
                    constants.MAX_ARMOR_COUNT,
                    constants.WEAPON_COUNT,
                ),
                -1.0e9,
                device=device,
                dtype=dtype,
            ),
            out_of_combat_s=torch.where(
                ground_mask,
                torch.full(
                    env_unit,
                    constants.OUT_OF_COMBAT_S,
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros(env_unit, device=device, dtype=dtype),
            ),
            respawn_progress=torch.zeros(env_unit, device=device, dtype=dtype),
            respawn_required=torch.zeros(env_unit, device=device, dtype=dtype),
            immediate_respawn_count=torch.zeros(
                env_unit,
                device=device,
                dtype=torch.long,
            ),
            invulnerable_s=torch.zeros(env_unit, device=device, dtype=dtype),
            weak=torch.zeros(env_unit, device=device, dtype=torch.bool),
            team_coin=torch.full(
                env_team,
                constants.INITIAL_TEAM_COIN,
                device=device,
                dtype=torch.long,
            ),
            team_purchased_ammo=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.WEAPON_COUNT),
                device=device,
                dtype=torch.long,
            ),
            team_damage=torch.zeros(env_team, device=device, dtype=dtype),
            base_shield=torch.full(
                env_team,
                constants.BASE_SHIELD,
                device=device,
                dtype=dtype,
            ),
            base_armor_deployed=torch.zeros(
                env_team,
                device=device,
                dtype=torch.bool,
            ),
            outpost_destroyed_once=torch.zeros(
                env_team,
                device=device,
                dtype=torch.bool,
            ),
            hero_42_silence_s=torch.zeros(env_team, device=device, dtype=dtype),
            hero_42_invalid_block=torch.zeros(
                env_team,
                device=device,
                dtype=torch.bool,
            ),
            radar_x=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                device=device,
                dtype=dtype,
            ),
            radar_p=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                device=device,
                dtype=dtype,
            ),
            radar_last_quality=torch.full(
                (num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                RadarQuality.UNKNOWN,
                device=device,
                dtype=torch.int8,
            ),
            elapsed_s=torch.zeros(num_envs, device=device, dtype=dtype),
            done=torch.zeros(num_envs, device=device, dtype=torch.bool),
            winner=torch.full(
                (num_envs,),
                Winner.UNDECIDED,
                device=device,
                dtype=torch.int8,
            ),
        )

    @property
    def num_envs(self) -> int:
        return int(self.hp.shape[0])

    @property
    def device(self) -> torch.device:
        return self.hp.device

    @property
    def dtype(self) -> torch.dtype:
        return self.hp.dtype

    def clone(self) -> "GameState":
        return GameState(
            **{field.name: getattr(self, field.name).clone() for field in fields(self)}
        )

    def to(self, device: torch.device | str) -> "GameState":
        return GameState(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )

    def validate(self) -> None:
        expected = (self.num_envs, constants.UNIT_COUNT)
        if self.hp.shape != expected or self.max_hp.shape != expected:
            raise ValueError("hp and max_hp must have shape [env, unit]")
        if self.heat.shape != (*expected, constants.WEAPON_COUNT):
            raise ValueError("heat must have shape [env, unit, weapon]")
        if not torch.all(torch.isfinite(self.hp)):
            raise ValueError("hp contains non-finite values")
        if torch.any(self.hp < 0) or torch.any(self.hp > self.max_hp):
            raise ValueError("hp must remain within [0, max_hp]")
        if torch.any(self.heat < 0):
            raise ValueError("heat cannot be negative")
        if torch.any(self.ammo < 0):
            raise ValueError("ammunition cannot be negative")
        if torch.any(self.team_coin < 0):
            raise ValueError("team coin cannot be negative")
        if torch.any(self.power_buffer_j < 0) or torch.any(
            self.power_buffer_j > constants.POWER_BUFFER_MAX_J
        ):
            raise ValueError("power buffer is outside [0, 60] J")
