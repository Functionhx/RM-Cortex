"""Leakage-safe, read-only access to the RMUC referee-system SQLite export.

The export contains 1 Hz state snapshots and referee events, not low-level
controls. This module therefore exposes tactical labels (future displacement
and whether a robot fired) while retaining explicit quality masks for missing
or invalid source values.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import sqlite3
from typing import Literal
from urllib.parse import quote

import torch
from torch import Tensor
from torch.utils.data import Dataset


SplitName = Literal["train", "validation", "test"]

RMUC_ROBOT_IDS = (1, 2, 3, 4, 6, 7, 10, 11, 101, 102, 103, 104, 106, 107, 110, 111)
RMUC_FEATURE_NAMES = (
    "hp",
    "max_hp",
    "x",
    "y",
    "z",
    "turret_yaw",
    "chassis_power",
    "heat_17mm",
    "heat_limit_17mm",
    "heat_42mm",
    "heat_limit_42mm",
    "shots_17mm_cumulative",
    "shots_42mm_cumulative",
    "team_coin_total",
    "team_coin_remaining",
    "vulnerable",
)

_FEATURE_COLUMNS = (
    "当前血量",
    "最大血量",
    "x",
    "y",
    "z",
    "枪口朝向",
    "底盘功率",
    "小热量",
    "小热量上限",
    "大热量",
    "大热量上限",
    "累计17mm发弹",
    "累计42mm发弹",
    "队伍总金币",
    "队伍剩余金币",
    "是否易伤",
)
_NONNEGATIVE_FEATURES = frozenset(
    {
        "hp",
        "max_hp",
        "heat_17mm",
        "heat_limit_17mm",
        "heat_42mm",
        "heat_limit_42mm",
        "shots_17mm_cumulative",
        "shots_42mm_cumulative",
        "team_coin_total",
        "team_coin_remaining",
    }
)
_WEAPON_ROLE_IDS = frozenset({1, 3, 4, 6, 7})
_MOBILE_ROLE_IDS = frozenset({1, 2, 3, 4, 6, 7})
_POSITION_FEATURE_INDICES = (RMUC_FEATURE_NAMES.index("x"), RMUC_FEATURE_NAMES.index("y"))
_HP_FEATURE_INDEX = RMUC_FEATURE_NAMES.index("hp")
_FIELD_X_RANGE_M = (0.0, 28.0)
_FIELD_Y_RANGE_M = (0.0, 15.0)
_MAX_INTENT_SPEED_MPS = 4.0

_REQUIRED_COLUMNS = {
    "matches": {
        "赛区",
        "场次号",
        "赛程",
        "局号",
        "game_id",
        "红方学校",
        "蓝方学校",
        "开始时间",
        "时长秒",
    },
    "timeseries": {
        "game_id",
        "时刻秒",
        "robot_id",
        "机器人类型",
        "阵营",
        "学校名",
        *_FEATURE_COLUMNS,
    },
    "events": {
        "game_id",
        "时刻秒",
        "事件类型",
        "robot_id",
        "类别",
    },
}


class RMUCSchemaError(ValueError):
    """Raised when an input database is not the supported RMUC export."""


@dataclass(frozen=True)
class MatchRecord:
    """Metadata used to keep windows inside one match and split by provenance."""

    game_id: int
    region: str
    match_number: int
    schedule: str
    round_number: int
    red_team: str
    blue_team: str
    start_time: datetime
    duration_s: float


@dataclass(frozen=True)
class SplitComponent:
    """A matchup-graph component assigned atomically to one temporal split."""

    component_id: str
    split: SplitName
    teams: tuple[str, ...]
    game_ids: tuple[int, ...]
    first_start: datetime
    last_start: datetime


@dataclass(frozen=True)
class TeamTimeSplitManifest:
    """Immutable split metadata with team-disjoint and chronological partitions."""

    assignments: Mapping[int, SplitName]
    components: tuple[SplitComponent, ...]

    def split_for(self, game_id: int) -> SplitName:
        try:
            return self.assignments[game_id]
        except KeyError as error:
            raise KeyError(f"game_id {game_id} is absent from the split manifest") from error

    def game_ids(self, split: SplitName) -> tuple[int, ...]:
        return tuple(
            sorted(game_id for game_id, value in self.assignments.items() if value == split)
        )


@dataclass(frozen=True)
class TacticalWindowConfig:
    """Sampling geometry for 1 Hz referee-system sequences."""

    history_steps: int = 10
    future_horizon_s: float = 5.0
    stride_s: float = 1.0
    sample_period_s: float = 1.0
    first_sample_s: float = 1.0

    def validate(self) -> None:
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive")
        if self.future_horizon_s <= 0.0:
            raise ValueError("future_horizon_s must be positive")
        if self.stride_s <= 0.0 or self.sample_period_s <= 0.0:
            raise ValueError("stride_s and sample_period_s must be positive")
        for name, value in (
            ("future_horizon_s", self.future_horizon_s),
            ("stride_s", self.stride_s),
            ("first_sample_s", self.first_sample_s),
        ):
            steps = value / self.sample_period_s
            if not math.isclose(steps, round(steps), abs_tol=1.0e-6):
                raise ValueError(f"{name} must be an integer multiple of sample_period_s")


@dataclass(frozen=True)
class TacticalSampleKey:
    """Stable identity for one team-perspective tactical decision."""

    game_id: int
    team: int
    anchor_s: float

    def __post_init__(self) -> None:
        if self.team not in (0, 1):
            raise ValueError("team must be 0 (red) or 1 (blue)")


@dataclass(frozen=True)
class TacticalSample:
    """A masked history and labels derived without low-level action assumptions.

    ``fired`` means that at least one referee firing event occurs in
    ``(anchor_s, anchor_s + future_horizon_s]``. It is a horizon-level occurrence
    proxy, not a claim that an instantaneous trigger or target choice is known.
    """

    key: TacticalSampleKey
    times_s: Tensor
    robot_ids: Tensor
    features: Tensor
    feature_valid: Tensor
    entity_present: Tensor
    row_unique: Tensor
    position_valid: Tensor
    own_entity: Tensor
    controllable: Tensor
    future_displacement_xy: Tensor
    future_displacement_valid: Tensor
    fired: Tensor
    fire_valid: Tensor


@dataclass(frozen=True)
class TacticalBatch:
    """Tensor-only collation of tactical samples for a training step."""

    game_id: Tensor
    team: Tensor
    anchor_s: Tensor
    times_s: Tensor
    robot_ids: Tensor
    features: Tensor
    feature_valid: Tensor
    entity_present: Tensor
    row_unique: Tensor
    position_valid: Tensor
    own_entity: Tensor
    controllable: Tensor
    future_displacement_xy: Tensor
    future_displacement_valid: Tensor
    fired: Tensor
    fire_valid: Tensor


class RMUCSQLiteAdapter:
    """Validate and query an RMUC SQLite export without ever opening it writable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._uri = f"file:{quote(self.path.as_posix(), safe='/')}?mode=ro"
        self._validate_schema()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a mode=ro/query_only connection and close it deterministically."""

        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    def _validate_schema(self) -> None:
        with self.connect() as connection:
            tables = {
                str(row["name"])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            missing_tables = sorted(_REQUIRED_COLUMNS.keys() - tables)
            if missing_tables:
                raise RMUCSchemaError(f"unsupported RMUC database: missing tables {missing_tables}")
            missing_by_table: dict[str, list[str]] = {}
            for table, required in _REQUIRED_COLUMNS.items():
                columns = {
                    str(row["name"]) for row in connection.execute(f'PRAGMA table_info("{table}")')
                }
                missing = sorted(required - columns)
                if missing:
                    missing_by_table[table] = missing
            if missing_by_table:
                details = "; ".join(
                    f"{table}: {columns}" for table, columns in missing_by_table.items()
                )
                raise RMUCSchemaError(f"unsupported RMUC database; missing columns: {details}")

    def matches(self) -> tuple[MatchRecord, ...]:
        """Return matches ordered by start time, rejecting ambiguous metadata."""

        query = """
            SELECT game_id, 赛区, 场次号, 赛程, 局号, 红方学校, 蓝方学校,
                   开始时间, 时长秒
            FROM matches
            ORDER BY 开始时间, game_id
        """
        with self.connect() as connection:
            rows = tuple(connection.execute(query))
        records = tuple(self._parse_match(row) for row in rows)
        game_ids = [record.game_id for record in records]
        if len(game_ids) != len(set(game_ids)):
            raise RMUCSchemaError("matches.game_id must be unique")
        return records

    @staticmethod
    def _parse_match(row: sqlite3.Row) -> MatchRecord:
        required_text = ("赛区", "赛程", "红方学校", "蓝方学校", "开始时间")
        for column in required_text:
            value = row[column]
            if not isinstance(value, str) or not value.strip():
                raise RMUCSchemaError(f"matches.{column} must be non-empty")
        try:
            start_time = datetime.fromisoformat(str(row["开始时间"]))
            duration_s = float(row["时长秒"])
            game_id = int(row["game_id"])
            match_number = int(row["场次号"])
            round_number = int(row["局号"])
        except (TypeError, ValueError) as error:
            raise RMUCSchemaError("invalid numeric or timestamp match metadata") from error
        if duration_s <= 0.0:
            raise RMUCSchemaError("matches.时长秒 must be positive")
        return MatchRecord(
            game_id=game_id,
            region=str(row["赛区"]).strip(),
            match_number=match_number,
            schedule=str(row["赛程"]).strip(),
            round_number=round_number,
            red_team=str(row["红方学校"]).strip(),
            blue_team=str(row["蓝方学校"]).strip(),
            start_time=start_time,
            duration_s=duration_s,
        )

    def build_team_time_split(
        self,
        *,
        train_fraction: float = 0.7,
        validation_fraction: float = 0.15,
    ) -> TeamTimeSplitManifest:
        """Build strict splits from chronological matchup-graph components.

        Every team and all of its opponents form a connected component. Components
        are assigned atomically, so no school occurs in multiple splits. Components
        are ordered by their complete time ranges; overlapping ranges are rejected
        instead of silently weakening temporal isolation. The regional 2026 export
        naturally forms three non-overlapping components.
        """

        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if train_fraction + validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be below 1")

        matches = self.matches()
        components = _matchup_components(matches)
        if len(components) < 3:
            raise RMUCSchemaError(
                "strict train/validation/test isolation requires at least three "
                "team-disjoint matchup components"
            )
        components.sort(key=lambda item: (item[1][0].start_time, item[0]))
        for previous, following in zip(components, components[1:], strict=False):
            previous_last = max(match.start_time for match in previous[1])
            following_first = min(match.start_time for match in following[1])
            if previous_last >= following_first:
                raise RMUCSchemaError(
                    "team-disjoint matchup components overlap in time; a strict "
                    "team/time split cannot be constructed"
                )

        train_count, validation_count = _partition_counts(
            len(components), train_fraction, validation_fraction
        )
        split_components: list[SplitComponent] = []
        assignments: dict[int, SplitName] = {}
        for index, (teams, component_matches) in enumerate(components):
            if index < train_count:
                split: SplitName = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            ordered_matches = sorted(
                component_matches, key=lambda item: (item.start_time, item.game_id)
            )
            component_id = hashlib.sha256("\0".join(teams).encode()).hexdigest()[:16]
            game_ids = tuple(match.game_id for match in ordered_matches)
            split_components.append(
                SplitComponent(
                    component_id=component_id,
                    split=split,
                    teams=teams,
                    game_ids=game_ids,
                    first_start=ordered_matches[0].start_time,
                    last_start=ordered_matches[-1].start_time,
                )
            )
            assignments.update({game_id: split for game_id in game_ids})
        return TeamTimeSplitManifest(assignments=assignments, components=tuple(split_components))

    def load_tactical_sample(
        self,
        key: TacticalSampleKey,
        config: TacticalWindowConfig | None = None,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> TacticalSample:
        """Load one masked team-perspective sample.

        Passing a connection is intended for :class:`RMUCTacticalDataset`; it must
        have been opened by this adapter so read-only guarantees remain intact.
        """

        window = config or TacticalWindowConfig()
        window.validate()
        if connection is not None:
            return self._load_tactical_sample(connection, key, window)
        with self.connect() as owned_connection:
            return self._load_tactical_sample(owned_connection, key, window)

    def _load_tactical_sample(
        self,
        connection: sqlite3.Connection,
        key: TacticalSampleKey,
        config: TacticalWindowConfig,
    ) -> TacticalSample:
        history_start = key.anchor_s - (config.history_steps - 1) * config.sample_period_s
        future_s = key.anchor_s + config.future_horizon_s
        match_row = connection.execute(
            "SELECT 时长秒 FROM matches WHERE game_id = ?", (key.game_id,)
        ).fetchone()
        if match_row is None:
            raise KeyError(f"unknown game_id {key.game_id}")
        duration_s = float(match_row["时长秒"])
        if history_start < config.first_sample_s - 1.0e-6:
            raise ValueError("sample history begins before the first recorded second")
        if future_s > duration_s - config.sample_period_s + 1.0e-6:
            raise ValueError("sample future horizon extends beyond recorded match data")

        feature_columns = ", ".join(f'"{column}"' for column in _FEATURE_COLUMNS)
        rows = connection.execute(
            f"""
                SELECT 时刻秒, robot_id, {feature_columns}
                FROM timeseries
                WHERE game_id = ? AND 时刻秒 >= ? AND 时刻秒 <= ?
                ORDER BY 时刻秒, robot_id
            """,
            (key.game_id, history_start, future_s),
        )
        events = connection.execute(
            """
                SELECT robot_id
                FROM events
                WHERE game_id = ? AND 事件类型 = '发弹'
                  AND 时刻秒 > ? AND 时刻秒 <= ?
            """,
            (key.game_id, key.anchor_s, future_s),
        )
        return _assemble_sample(key, config, tuple(rows), tuple(events))


class RMUCTacticalDataset(Dataset[TacticalSample]):
    """Lazy split view that keeps one read-only SQLite connection per worker."""

    def __init__(
        self,
        adapter: RMUCSQLiteAdapter,
        manifest: TeamTimeSplitManifest,
        split: SplitName,
        config: TacticalWindowConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.manifest = manifest
        self.split = split
        self.config = config or TacticalWindowConfig()
        self.config.validate()
        matches = {match.game_id: match for match in adapter.matches()}
        missing = set(manifest.assignments) - matches.keys()
        if missing:
            raise ValueError(f"split manifest contains unknown game_ids: {sorted(missing)}")
        self.keys = tuple(
            TacticalSampleKey(match.game_id, team, anchor_s)
            for game_id in manifest.game_ids(split)
            for match in (matches[game_id],)
            for anchor_s in _sample_anchors(match.duration_s, self.config)
            for team in (0, 1)
        )
        self._connection: sqlite3.Connection | None = None
        self._connection_pid: int | None = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> TacticalSample:
        return self.adapter.load_tactical_sample(
            self.keys[index],
            self.config,
            connection=self._worker_connection(),
        )

    def _worker_connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._connection is None or self._connection_pid != pid:
            self.close()
            self._connection = self.adapter._open_connection()
            self._connection_pid = pid
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._connection_pid = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_connection_pid"] = None
        return state


def collate_tactical_samples(samples: Sequence[TacticalSample]) -> TacticalBatch:
    """Stack samples without teaching PyTorch's generic collator about dataclasses."""

    if not samples:
        raise ValueError("cannot collate an empty tactical batch")
    return TacticalBatch(
        game_id=torch.tensor(tuple(sample.key.game_id for sample in samples), dtype=torch.int64),
        team=torch.tensor(tuple(sample.key.team for sample in samples), dtype=torch.int64),
        anchor_s=torch.tensor(
            tuple(sample.key.anchor_s for sample in samples), dtype=torch.float32
        ),
        times_s=torch.stack(tuple(sample.times_s for sample in samples)),
        robot_ids=torch.stack(tuple(sample.robot_ids for sample in samples)),
        features=torch.stack(tuple(sample.features for sample in samples)),
        feature_valid=torch.stack(tuple(sample.feature_valid for sample in samples)),
        entity_present=torch.stack(tuple(sample.entity_present for sample in samples)),
        row_unique=torch.stack(tuple(sample.row_unique for sample in samples)),
        position_valid=torch.stack(tuple(sample.position_valid for sample in samples)),
        own_entity=torch.stack(tuple(sample.own_entity for sample in samples)),
        controllable=torch.stack(tuple(sample.controllable for sample in samples)),
        future_displacement_xy=torch.stack(
            tuple(sample.future_displacement_xy for sample in samples)
        ),
        future_displacement_valid=torch.stack(
            tuple(sample.future_displacement_valid for sample in samples)
        ),
        fired=torch.stack(tuple(sample.fired for sample in samples)),
        fire_valid=torch.stack(tuple(sample.fire_valid for sample in samples)),
    )


def _matchup_components(
    matches: Sequence[MatchRecord],
) -> list[tuple[tuple[str, ...], list[MatchRecord]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        if match.red_team == match.blue_team:
            raise RMUCSchemaError(f"game_id {match.game_id} has identical red and blue teams")
        adjacency[match.red_team].add(match.blue_team)
        adjacency[match.blue_team].add(match.red_team)

    team_components: list[tuple[str, ...]] = []
    visited: set[str] = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        pending = [root]
        visited.add(root)
        component: list[str] = []
        while pending:
            team = pending.pop()
            component.append(team)
            for opponent in sorted(adjacency[team], reverse=True):
                if opponent not in visited:
                    visited.add(opponent)
                    pending.append(opponent)
        team_components.append(tuple(sorted(component)))

    component_by_team = {team: component for component in team_components for team in component}
    matches_by_component: dict[tuple[str, ...], list[MatchRecord]] = defaultdict(list)
    for match in matches:
        matches_by_component[component_by_team[match.red_team]].append(match)
    return [(teams, matches_by_component[teams]) for teams in team_components]


def _partition_counts(
    component_count: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int]:
    train_count = max(1, int(component_count * train_fraction))
    validation_count = max(1, int(component_count * validation_fraction))
    while train_count + validation_count >= component_count:
        if train_count > validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise RMUCSchemaError("cannot allocate non-empty train/validation/test splits")
    return train_count, validation_count


def _sample_anchors(duration_s: float, config: TacticalWindowConfig) -> tuple[float, ...]:
    first = config.first_sample_s + (config.history_steps - 1) * config.sample_period_s
    last = duration_s - config.sample_period_s - config.future_horizon_s
    if last < first - 1.0e-6:
        return ()
    count = math.floor((last - first) / config.stride_s + 1.0e-6) + 1
    return tuple(first + index * config.stride_s for index in range(count))


def _perspective_robot_ids(team: int) -> tuple[int, ...]:
    return RMUC_ROBOT_IDS if team == 0 else RMUC_ROBOT_IDS[8:] + RMUC_ROBOT_IDS[:8]


def _feature_quality(name: str, value: object) -> tuple[float, bool]:
    if not isinstance(value, (int, float)):
        return 0.0, False
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0, False
    if name in _NONNEGATIVE_FEATURES and numeric < 0.0:
        return numeric, False
    if name == "vulnerable" and numeric not in (0.0, 1.0):
        return numeric, False
    return numeric, True


def _assemble_sample(
    key: TacticalSampleKey,
    config: TacticalWindowConfig,
    rows: Sequence[sqlite3.Row],
    event_rows: Sequence[sqlite3.Row],
) -> TacticalSample:
    robot_ids = _perspective_robot_ids(key.team)
    slot_by_id = {robot_id: slot for slot, robot_id in enumerate(robot_ids)}
    history_start = key.anchor_s - (config.history_steps - 1) * config.sample_period_s
    expected_times = tuple(
        history_start + index * config.sample_period_s for index in range(config.history_steps)
    )
    future_s = key.anchor_s + config.future_horizon_s
    all_times = (*expected_times, future_s)
    time_to_index = {round(value, 6): index for index, value in enumerate(all_times)}

    feature_shape = (len(all_times), len(robot_ids), len(RMUC_FEATURE_NAMES))
    all_features = torch.zeros(feature_shape, dtype=torch.float32)
    all_feature_valid = torch.zeros(feature_shape, dtype=torch.bool)
    all_present = torch.zeros((len(all_times), len(robot_ids)), dtype=torch.bool)
    all_unique = torch.zeros((len(all_times), len(robot_ids)), dtype=torch.bool)

    for row in rows:
        robot_id = int(row["robot_id"])
        slot = slot_by_id.get(robot_id)
        time_index = time_to_index.get(round(float(row["时刻秒"]), 6))
        if slot is None or time_index is None:
            continue
        if all_present[time_index, slot]:
            all_unique[time_index, slot] = False
            all_feature_valid[time_index, slot] = False
            continue
        all_present[time_index, slot] = True
        all_unique[time_index, slot] = True
        for feature_index, (name, column) in enumerate(
            zip(RMUC_FEATURE_NAMES, _FEATURE_COLUMNS, strict=True)
        ):
            value, valid = _feature_quality(name, row[column])
            all_features[time_index, slot, feature_index] = value
            all_feature_valid[time_index, slot, feature_index] = valid

    all_feature_valid &= all_unique[..., None]
    role_ids = torch.tensor(tuple(robot_id % 100 for robot_id in robot_ids))
    x_index, y_index = _POSITION_FEATURE_INDICES
    x = all_features[..., x_index]
    y = all_features[..., y_index]
    numeric_xy = all_feature_valid[..., x_index] & all_feature_valid[..., y_index]
    non_origin = ~((x == 0.0) & (y == 0.0))
    dynamic_position_valid = (
        numeric_xy
        & non_origin
        & (x >= _FIELD_X_RANGE_M[0])
        & (x <= _FIELD_X_RANGE_M[1])
        & (y >= _FIELD_Y_RANGE_M[0])
        & (y <= _FIELD_Y_RANGE_M[1])
    )
    mobile = torch.tensor(
        tuple(role_id.item() in _MOBILE_ROLE_IDS for role_id in role_ids),
        dtype=torch.bool,
    )
    # Static structures in this export use an unusable (0, 0, 0) placeholder.
    # Reject it without inventing coordinates; upper layers may inject official
    # structure coordinates explicitly.
    position_valid = torch.where(
        mobile[None, :],
        dynamic_position_valid,
        numeric_xy & non_origin,
    )
    all_feature_valid[..., x_index] &= position_valid
    all_feature_valid[..., y_index] &= position_valid

    anchor_index = config.history_steps - 1
    future_index = config.history_steps
    future_displacement = (
        all_features[future_index, :, _POSITION_FEATURE_INDICES]
        - all_features[anchor_index, :, _POSITION_FEATURE_INDICES]
    )

    own_entity = torch.zeros(len(robot_ids), dtype=torch.bool)
    own_entity[:8] = True
    controllable = own_entity & mobile
    alive_for_label = all_feature_valid[..., _HP_FEATURE_INDEX] & (
        all_features[..., _HP_FEATURE_INDEX] > 0.0
    )
    displacement_within_limit = (
        torch.linalg.vector_norm(future_displacement, dim=-1)
        <= _MAX_INTENT_SPEED_MPS * config.future_horizon_s + 1.0e-6
    )
    future_displacement_valid = (
        controllable
        & position_valid[anchor_index]
        & position_valid[future_index]
        & alive_for_label[anchor_index]
        & alive_for_label[future_index]
        & displacement_within_limit
    )

    fired = torch.zeros(len(robot_ids), dtype=torch.bool)
    for event in event_rows:
        slot = slot_by_id.get(int(event["robot_id"]))
        if slot is not None:
            fired[slot] = True
    weapon_capable = torch.tensor(
        tuple(role_id.item() in _WEAPON_ROLE_IDS for role_id in role_ids),
        dtype=torch.bool,
    )
    fire_valid = (
        own_entity
        & weapon_capable
        & all_present[anchor_index]
        & all_unique[anchor_index]
        & position_valid[anchor_index]
        & alive_for_label[anchor_index]
    )

    return TacticalSample(
        key=key,
        times_s=torch.tensor(expected_times, dtype=torch.float32),
        robot_ids=torch.tensor(robot_ids, dtype=torch.int64),
        features=all_features[: config.history_steps],
        feature_valid=all_feature_valid[: config.history_steps],
        entity_present=all_present[: config.history_steps],
        row_unique=all_unique[: config.history_steps],
        position_valid=position_valid[: config.history_steps],
        own_entity=own_entity,
        controllable=controllable,
        future_displacement_xy=future_displacement,
        future_displacement_valid=future_displacement_valid,
        fired=fired,
        fire_valid=fire_valid,
    )
