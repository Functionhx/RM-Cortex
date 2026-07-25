"""Vectorized authoritative game state."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import (
    DartGateState,
    HeatLock,
    InfantryChassisProfile,
    InfantryLauncherProfile,
    RadarQuality,
    Role,
    RuneMode,
    SentryStance,
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
    hero_profile: Tensor
    infantry_chassis_profile: Tensor
    infantry_launcher_profile: Tensor

    heat: Tensor
    heat_limit: Tensor
    base_cooling_per_s: Tensor
    cooling_per_s: Tensor
    heat_lock: Tensor
    video_disabled: Tensor
    ammo: Tensor
    muzzle_velocity_limit_mps: Tensor
    velocity_lock_s: Tensor
    velocity_permanent_lock: Tensor
    speed_module_offline: Tensor

    base_power_limit_w: Tensor
    power_limit_w: Tensor
    power_buffer_j: Tensor
    chassis_disabled_s: Tensor
    chassis_energy: Tensor
    immediate_power_boost_s: Tensor

    controller_offline: Tensor
    controller_offline_s: Tensor
    offline_module_accumulator_s: Tensor

    attack_multiplier: Tensor
    defense_fraction: Tensor
    vulnerability_fraction: Tensor
    radar_vulnerability_fraction: Tensor
    role_defense_fraction: Tensor
    role_vulnerability_fraction: Tensor
    role_cooling_multiplier: Tensor
    role_power_multiplier: Tensor
    zone_defense_fraction: Tensor
    zone_vulnerability_fraction: Tensor
    terrain_defense_fraction: Tensor
    zone_cooling_bonus_per_s: Tensor
    fortress_reserve_ammo: Tensor
    fortress_reserve_cap: Tensor

    armor_last_hit_s: Tensor
    out_of_combat_s: Tensor
    dead_s: Tensor
    respawn_progress: Tensor
    respawn_required: Tensor
    immediate_respawn_count: Tensor
    invulnerable_s: Tensor
    weak: Tensor
    remote_heal_active: Tensor
    remote_heal_progress_s: Tensor

    team_coin: Tensor
    team_total_coin_earned: Tensor
    team_purchased_ammo: Tensor
    team_damage: Tensor
    pending_purchase_unit: Tensor
    pending_purchase_weapon: Tensor
    pending_purchase_kind: Tensor
    pending_purchase_progress_s: Tensor
    sentry_supply_rounds: Tensor

    team_level_cap: Tensor
    tech_completion_count: Tensor
    tech_income_per_10s: Tensor
    tech_defense_fraction: Tensor
    tech_energy_units_remaining: Tensor

    base_shield: Tensor
    base_armor_deployed: Tensor
    base_hp_lost: Tensor
    outpost_destroyed_once: Tensor
    outpost_rebuild_charges: Tensor
    outpost_rebuild_awarded: Tensor
    outpost_rebuild_progress_s: Tensor
    outpost_angle_rad: Tensor
    outpost_speed_rad_s: Tensor
    outpost_rotation_direction: Tensor
    outpost_rotation_stopped: Tensor

    hero_42_silence_s: Tensor
    hero_42_invalid_block: Tensor
    hero_deploy_progress_s: Tensor
    hero_deployed: Tensor
    sentry_stance: Tensor
    sentry_stance_cooldown_s: Tensor
    sentry_stance_time_s: Tensor
    sentry_automatic: Tensor
    aerial_support_bank_s: Tensor
    aerial_support_active: Tensor
    aerial_on_pad: Tensor
    aerial_coin_accumulator_s: Tensor
    aerial_counter_p: Tensor
    aerial_counter_streak: Tensor
    aerial_counter_lock_s: Tensor
    aerial_counter_uses: Tensor

    rune_small_opportunities: Tensor
    rune_large_opportunities: Tensor
    rune_mode: Tensor
    rune_trigger_remaining_s: Tensor
    rune_group_remaining_s: Tensor
    rune_groups: Tensor
    rune_arms: Tensor
    rune_ring_sum: Tensor
    rune_ring_count: Tensor
    rune_buff_s: Tensor
    rune_small_buff_s: Tensor
    rune_attack_multiplier: Tensor
    rune_defense_fraction: Tensor
    rune_cooling_multiplier: Tensor
    rune_small_bonus_xp: Tensor

    dart_rounds: Tensor
    dart_gate_opportunities: Tensor
    dart_gate_state: Tensor
    dart_gate_timer_s: Tensor
    dart_detection_s: Tensor
    dart_detection_block_s: Tensor
    dart_zone_disabled_s: Tensor
    dart_hits_by_target: Tensor

    radar_x: Tensor
    radar_p: Tensor
    radar_last_quality: Tensor
    radar_no_data_s: Tensor
    radar_report_xy: Tensor
    radar_report_valid: Tensor
    radar_report_age_s: Tensor
    radar_truth_visible: Tensor
    radar_vulnerability_progress_s: Tensor
    radar_double_charges: Tensor
    radar_double_uses: Tensor
    radar_double_s: Tensor
    radar_interference_level: Tensor

    zone_linger_s: Tensor
    terrain_buff_s: Tensor
    terrain_seen: Tensor
    terrain_group_enhanced_s: Tensor
    terrain_road_cooldown_s: Tensor
    tunnel_cooling_s: Tensor
    assembly_invulnerable_used_s: Tensor
    enemy_fortress_progress_s: Tensor
    enemy_fortress_hold_s: Tensor

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
        default_velocity = torch.tensor(
            constants.DEFAULT_MUZZLE_VELOCITY_LIMIT_MPS * constants.TEAM_COUNT,
            device=device,
            dtype=dtype,
        )

        power_mask = chassis_power_mask(device).unsqueeze(0).expand(env_unit)
        energy_mask = chassis_energy_mask(device).unsqueeze(0).expand(env_unit)
        ground_mask = ground_robot_mask(device).unsqueeze(0).expand(env_unit)
        rotation_direction = torch.where(
            torch.arange(num_envs, device=device) % 2 == 0,
            torch.ones(num_envs, device=device, dtype=dtype),
            -torch.ones(num_envs, device=device, dtype=dtype),
        )

        return cls(
            hp=hp,
            max_hp=max_hp,
            alive=alive,
            level=torch.ones(env_unit, device=device, dtype=torch.long),
            xp=torch.zeros(env_unit, device=device, dtype=dtype),
            hero_profile=torch.zeros(env_unit, device=device, dtype=torch.int8),
            infantry_chassis_profile=torch.full(
                env_unit,
                InfantryChassisProfile.HP,
                device=device,
                dtype=torch.int8,
            ),
            infantry_launcher_profile=torch.full(
                env_unit,
                InfantryLauncherProfile.COOLING,
                device=device,
                dtype=torch.int8,
            ),
            heat=torch.zeros(env_weapon, device=device, dtype=dtype),
            heat_limit=default_heat_limit.unsqueeze(0).expand(env_weapon).clone(),
            base_cooling_per_s=default_cooling.unsqueeze(0).expand(env_weapon).clone(),
            cooling_per_s=default_cooling.unsqueeze(0).expand(env_weapon).clone(),
            heat_lock=torch.full(
                env_weapon,
                HeatLock.NONE,
                device=device,
                dtype=torch.int8,
            ),
            video_disabled=torch.zeros(env_unit, device=device, dtype=torch.bool),
            ammo=default_ammo.unsqueeze(0).expand(env_weapon).clone(),
            muzzle_velocity_limit_mps=default_velocity.unsqueeze(0).expand(env_weapon).clone(),
            velocity_lock_s=torch.zeros(env_weapon, device=device, dtype=dtype),
            velocity_permanent_lock=torch.zeros(env_weapon, device=device, dtype=torch.bool),
            speed_module_offline=torch.zeros(env_weapon, device=device, dtype=torch.bool),
            base_power_limit_w=default_power.unsqueeze(0).expand(env_unit).clone(),
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
            immediate_power_boost_s=torch.zeros(env_unit, device=device, dtype=dtype),
            controller_offline=torch.zeros(env_unit, device=device, dtype=torch.bool),
            controller_offline_s=torch.zeros(env_unit, device=device, dtype=dtype),
            offline_module_accumulator_s=torch.zeros(env_unit, device=device, dtype=dtype),
            attack_multiplier=torch.ones(env_unit, device=device, dtype=dtype),
            defense_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            vulnerability_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            radar_vulnerability_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            role_defense_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            role_vulnerability_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            role_cooling_multiplier=torch.ones(env_unit, device=device, dtype=dtype),
            role_power_multiplier=torch.ones(env_unit, device=device, dtype=dtype),
            zone_defense_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            zone_vulnerability_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            terrain_defense_fraction=torch.zeros(env_unit, device=device, dtype=dtype),
            zone_cooling_bonus_per_s=torch.zeros(env_weapon, device=device, dtype=dtype),
            fortress_reserve_ammo=torch.zeros(env_unit, device=device, dtype=torch.long),
            fortress_reserve_cap=torch.zeros(env_unit, device=device, dtype=torch.long),
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
                torch.full(env_unit, constants.OUT_OF_COMBAT_S, device=device, dtype=dtype),
                torch.zeros(env_unit, device=device, dtype=dtype),
            ),
            dead_s=torch.zeros(env_unit, device=device, dtype=dtype),
            respawn_progress=torch.zeros(env_unit, device=device, dtype=dtype),
            respawn_required=torch.zeros(env_unit, device=device, dtype=dtype),
            immediate_respawn_count=torch.zeros(env_unit, device=device, dtype=torch.long),
            invulnerable_s=torch.zeros(env_unit, device=device, dtype=dtype),
            weak=torch.zeros(env_unit, device=device, dtype=torch.bool),
            remote_heal_active=torch.zeros(env_unit, device=device, dtype=torch.bool),
            remote_heal_progress_s=torch.zeros(env_unit, device=device, dtype=dtype),
            team_coin=torch.full(
                env_team,
                constants.INITIAL_TEAM_COIN,
                device=device,
                dtype=torch.long,
            ),
            team_total_coin_earned=torch.full(
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
            pending_purchase_unit=torch.full(env_team, -1, device=device, dtype=torch.long),
            pending_purchase_weapon=torch.zeros(env_team, device=device, dtype=torch.long),
            pending_purchase_kind=torch.zeros(env_team, device=device, dtype=torch.int8),
            pending_purchase_progress_s=torch.zeros(env_team, device=device, dtype=dtype),
            sentry_supply_rounds=torch.zeros(env_team, device=device, dtype=torch.long),
            team_level_cap=torch.full(env_team, 5, device=device, dtype=torch.long),
            tech_completion_count=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.TECH_LEVEL_COUNT),
                device=device,
                dtype=torch.long,
            ),
            tech_income_per_10s=torch.zeros(env_team, device=device, dtype=torch.long),
            tech_defense_fraction=torch.zeros(env_team, device=device, dtype=dtype),
            tech_energy_units_remaining=torch.full(
                (num_envs,),
                constants.TECH_ENERGY_UNITS_INITIAL,
                device=device,
                dtype=torch.long,
            ),
            base_shield=torch.full(env_team, constants.BASE_SHIELD, device=device, dtype=dtype),
            base_armor_deployed=torch.zeros(env_team, device=device, dtype=torch.bool),
            base_hp_lost=torch.zeros(env_team, device=device, dtype=dtype),
            outpost_destroyed_once=torch.zeros(env_team, device=device, dtype=torch.bool),
            outpost_rebuild_charges=torch.zeros(env_team, device=device, dtype=torch.long),
            outpost_rebuild_awarded=torch.zeros(env_team, device=device, dtype=torch.long),
            outpost_rebuild_progress_s=torch.zeros(env_unit, device=device, dtype=dtype),
            outpost_angle_rad=torch.zeros(env_team, device=device, dtype=dtype),
            outpost_speed_rad_s=torch.zeros(env_team, device=device, dtype=dtype),
            outpost_rotation_direction=rotation_direction[:, None].expand(env_team).clone(),
            outpost_rotation_stopped=torch.zeros(env_team, device=device, dtype=torch.bool),
            hero_42_silence_s=torch.zeros(env_team, device=device, dtype=dtype),
            hero_42_invalid_block=torch.zeros(env_team, device=device, dtype=torch.bool),
            hero_deploy_progress_s=torch.zeros(env_team, device=device, dtype=dtype),
            hero_deployed=torch.zeros(env_team, device=device, dtype=torch.bool),
            sentry_stance=torch.full(
                env_team,
                SentryStance.MOBILE,
                device=device,
                dtype=torch.int8,
            ),
            sentry_stance_cooldown_s=torch.zeros(env_team, device=device, dtype=dtype),
            sentry_stance_time_s=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.SENTRY_STANCE_COUNT),
                device=device,
                dtype=dtype,
            ),
            sentry_automatic=torch.ones(env_team, device=device, dtype=torch.bool),
            aerial_support_bank_s=torch.full(
                env_team,
                constants.AERIAL_SUPPORT_INITIAL_S,
                device=device,
                dtype=dtype,
            ),
            aerial_support_active=torch.zeros(env_team, device=device, dtype=torch.bool),
            aerial_on_pad=torch.ones(env_team, device=device, dtype=torch.bool),
            aerial_coin_accumulator_s=torch.zeros(env_team, device=device, dtype=dtype),
            aerial_counter_p=torch.zeros(env_team, device=device, dtype=dtype),
            aerial_counter_streak=torch.zeros(env_team, device=device, dtype=torch.long),
            aerial_counter_lock_s=torch.zeros(env_team, device=device, dtype=dtype),
            aerial_counter_uses=torch.zeros(env_team, device=device, dtype=torch.long),
            rune_small_opportunities=torch.ones(env_team, device=device, dtype=torch.long),
            rune_large_opportunities=torch.zeros(env_team, device=device, dtype=torch.long),
            rune_mode=torch.full(env_team, RuneMode.IDLE, device=device, dtype=torch.int8),
            rune_trigger_remaining_s=torch.zeros(env_team, device=device, dtype=dtype),
            rune_group_remaining_s=torch.zeros(env_team, device=device, dtype=dtype),
            rune_groups=torch.zeros(env_team, device=device, dtype=torch.long),
            rune_arms=torch.zeros(env_team, device=device, dtype=torch.long),
            rune_ring_sum=torch.zeros(env_team, device=device, dtype=dtype),
            rune_ring_count=torch.zeros(env_team, device=device, dtype=torch.long),
            rune_buff_s=torch.zeros(env_team, device=device, dtype=dtype),
            rune_small_buff_s=torch.zeros(env_team, device=device, dtype=dtype),
            rune_attack_multiplier=torch.ones(env_team, device=device, dtype=dtype),
            rune_defense_fraction=torch.zeros(env_team, device=device, dtype=dtype),
            rune_cooling_multiplier=torch.ones(env_team, device=device, dtype=dtype),
            rune_small_bonus_xp=torch.zeros(env_team, device=device, dtype=dtype),
            dart_rounds=torch.full(
                env_team,
                constants.DART_ROUNDS_INITIAL,
                device=device,
                dtype=torch.long,
            ),
            dart_gate_opportunities=torch.zeros(env_team, device=device, dtype=torch.long),
            dart_gate_state=torch.full(
                env_team,
                DartGateState.CLOSED,
                device=device,
                dtype=torch.int8,
            ),
            dart_gate_timer_s=torch.zeros(env_team, device=device, dtype=dtype),
            dart_detection_s=torch.zeros(env_team, device=device, dtype=dtype),
            dart_detection_block_s=torch.zeros(env_team, device=device, dtype=dtype),
            dart_zone_disabled_s=torch.zeros(env_team, device=device, dtype=dtype),
            dart_hits_by_target=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.DART_TARGET_COUNT),
                device=device,
                dtype=torch.long,
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
            radar_no_data_s=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                device=device,
                dtype=dtype,
            ),
            radar_report_xy=torch.zeros(
                (
                    num_envs,
                    constants.TEAM_COUNT,
                    constants.UNIT_COUNT,
                    2,
                ),
                device=device,
                dtype=dtype,
            ),
            radar_report_valid=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                device=device,
                dtype=torch.bool,
            ),
            radar_report_age_s=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                device=device,
                dtype=dtype,
            ),
            radar_truth_visible=torch.zeros(
                (num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                device=device,
                dtype=torch.bool,
            ),
            radar_vulnerability_progress_s=torch.zeros(env_team, device=device, dtype=dtype),
            radar_double_charges=torch.zeros(env_team, device=device, dtype=torch.long),
            radar_double_uses=torch.zeros(env_team, device=device, dtype=torch.long),
            radar_double_s=torch.zeros(env_team, device=device, dtype=dtype),
            radar_interference_level=torch.ones(env_team, device=device, dtype=torch.long),
            zone_linger_s=torch.zeros(
                (num_envs, constants.UNIT_COUNT, constants.ZONE_COUNT),
                device=device,
                dtype=dtype,
            ),
            terrain_buff_s=torch.zeros(
                (num_envs, constants.UNIT_COUNT, constants.TERRAIN_COUNT),
                device=device,
                dtype=dtype,
            ),
            terrain_seen=torch.zeros(
                (num_envs, constants.UNIT_COUNT, constants.TERRAIN_COUNT),
                device=device,
                dtype=torch.bool,
            ),
            terrain_group_enhanced_s=torch.zeros(env_unit, device=device, dtype=dtype),
            terrain_road_cooldown_s=torch.zeros(env_unit, device=device, dtype=dtype),
            tunnel_cooling_s=torch.zeros(env_unit, device=device, dtype=dtype),
            assembly_invulnerable_used_s=torch.zeros(env_unit, device=device, dtype=dtype),
            enemy_fortress_progress_s=torch.zeros(env_unit, device=device, dtype=dtype),
            enemy_fortress_hold_s=torch.zeros(env_unit, device=device, dtype=dtype),
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
        if torch.any(self.chassis_energy < 0) or torch.any(
            self.chassis_energy > constants.CHASSIS_ENERGY_MAX
        ):
            raise ValueError("chassis energy is outside its official range")
        if torch.any(self.radar_p < 0) or torch.any(self.radar_p > 150):
            raise ValueError("radar progress is outside [0, 150]")
        radar_shape = (self.num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT)
        if self.radar_report_xy.shape != (*radar_shape, 2):
            raise ValueError("radar_report_xy must have shape [env, team, unit, 2]")
        if (
            self.radar_report_valid.shape != radar_shape
            or self.radar_report_age_s.shape != radar_shape
        ):
            raise ValueError("radar report metadata must have shape [env, team, unit]")
        if not torch.all(torch.isfinite(self.radar_report_age_s)) or torch.any(
            self.radar_report_age_s < 0
        ):
            raise ValueError("radar report age must be finite and non-negative")
        if not torch.all(torch.isfinite(self.radar_report_xy[self.radar_report_valid])):
            raise ValueError("valid radar reports must contain finite coordinates")
        if torch.any(self.team_level_cap < 1) or torch.any(self.team_level_cap > 10):
            raise ValueError("team level cap is outside [1, 10]")
