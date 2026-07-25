from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest
import torch

from rm_train.data import (
    RMUC_FEATURE_NAMES,
    RMUCSQLiteAdapter,
    RMUCSchemaError,
    RMUCTacticalDataset,
    TacticalSampleKey,
    TacticalWindowConfig,
    collate_tactical_samples,
)


@contextmanager
def _database(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE matches(
          赛区 TEXT, 场次号 INT, 赛程 TEXT, 局号 INT, game_id INT, web_game_id INT,
          红方学校 TEXT, 蓝方学校 TEXT, 胜方 TEXT, 开始时间 TEXT, 时长秒 INT
        );
        CREATE TABLE timeseries(
          赛区 TEXT, 场次号 INT, 赛程 TEXT, 局号 INT, game_id INT, 时刻秒 REAL,
          robot_id INT, 机器人类型 TEXT, 阵营 TEXT, 学校名 TEXT, 对手学校 TEXT,
          当前血量 REAL, 最大血量 REAL, x REAL, y REAL, z REAL, 枪口朝向 REAL,
          底盘功率 REAL, 小热量 REAL, 小热量上限 REAL, 大热量 REAL, 大热量上限 REAL,
          累计17mm发弹 REAL, 累计42mm发弹 REAL, 队伍总金币 REAL,
          队伍剩余金币 REAL, 是否易伤 INT
        );
        CREATE TABLE events(
          赛区 TEXT, 场次号 INT, 赛程 TEXT, 局号 INT, game_id INT, 时刻秒 REAL,
          事件类型 TEXT, robot_id INT, 机器人类型 TEXT, 阵营 TEXT, 学校名 TEXT,
          目标robot_id INT, 目标类型 TEXT, 类别 TEXT, 数值 REAL, 备注 TEXT
        );
        """
    )


def _insert_match(
    connection: sqlite3.Connection,
    game_id: int,
    red_team: str,
    blue_team: str,
    start_time: str,
    *,
    duration_s: int = 10,
) -> None:
    connection.execute(
        """
        INSERT INTO matches(
          赛区, 场次号, 赛程, 局号, game_id, web_game_id,
          红方学校, 蓝方学校, 胜方, 开始时间, 时长秒
        ) VALUES ('fixture', ?, 'fixture', 1, ?, NULL, ?, ?, '红', ?, ?)
        """,
        (game_id, game_id, red_team, blue_team, start_time, duration_s),
    )


def _insert_timeseries(
    connection: sqlite3.Connection,
    game_id: int,
    time_s: float,
    robot_id: int,
    *,
    hp: float | None = 100.0,
    x: float | None = 0.0,
    y: float | None = 0.0,
    shots_17: float | None = 0.0,
    shots_42: float | None = 0.0,
    vulnerable: int | None = 0,
) -> None:
    team = "红" if robot_id < 100 else "蓝"
    school = "A" if team == "红" else "B"
    connection.execute(
        """
        INSERT INTO timeseries VALUES(
          'fixture', 1, 'fixture', 1, ?, ?, ?, 'fixture', ?, ?, 'opponent',
          ?, 200, ?, ?, 0, 0, 20, 0, 100, 0, 100, ?, ?, 400, 300, ?
        )
        """,
        (
            game_id,
            time_s,
            robot_id,
            team,
            school,
            hp,
            x,
            y,
            shots_17,
            shots_42,
            vulnerable,
        ),
    )


def _insert_fire(
    connection: sqlite3.Connection,
    game_id: int,
    time_s: float,
    robot_id: int,
) -> None:
    connection.execute(
        """
        INSERT INTO events VALUES(
          'fixture', 1, 'fixture', 1, ?, ?, '发弹', ?, 'fixture',
          '红', 'A', NULL, NULL, '17mm', NULL, NULL
        )
        """,
        (game_id, time_s, robot_id),
    )


@pytest.fixture
def rmuc_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "rmuc fixture.sqlite"
    with _database(path) as connection:
        _create_schema(connection)
        _insert_match(connection, 1, "A", "B", "2026-01-01 10:00:00")
        _insert_match(connection, 2, "C", "D", "2026-01-02 10:00:00")
        _insert_match(connection, 3, "E", "F", "2026-01-03 10:00:00")

        _insert_timeseries(connection, 1, 2.0, 1, x=1.0, y=1.0, shots_42=None)
        _insert_timeseries(connection, 1, 4.0, 1, x=4.0, y=1.0, shots_17=-5.0, shots_42=2.0)
        _insert_timeseries(connection, 1, 6.0, 1, x=7.0, y=5.0, shots_42=3.0)
        _insert_timeseries(connection, 1, 2.0, 2, x=2.0, y=1.0)
        _insert_timeseries(connection, 1, 3.0, 2, x=3.0, y=1.0)
        _insert_timeseries(connection, 1, 4.0, 2, x=0.0, y=0.0)
        _insert_timeseries(connection, 1, 6.0, 2, x=6.0, y=1.0)
        _insert_timeseries(connection, 1, 4.0, 3, x=4.0, y=2.0)
        _insert_timeseries(connection, 1, 4.0, 3, x=40.0, y=20.0)
        _insert_timeseries(connection, 1, 4.0, 4, x=1.0, y=1.0)
        _insert_timeseries(connection, 1, 6.0, 4, x=10.0, y=1.0)
        _insert_timeseries(connection, 1, 4.0, 6, x=27.0, y=14.0)
        _insert_timeseries(connection, 1, 6.0, 6, x=29.0, y=14.0)
        _insert_timeseries(connection, 1, 4.0, 7, hp=0.0, x=2.0, y=2.0)
        _insert_timeseries(connection, 1, 6.0, 7, hp=100.0, x=3.0, y=2.0)
        _insert_timeseries(connection, 1, 4.0, 10, x=0.0, y=0.0)
        _insert_timeseries(connection, 1, 6.0, 10, x=0.0, y=0.0)
        _insert_timeseries(connection, 1, 2.0, 101, x=9.0, y=1.0)
        _insert_timeseries(connection, 1, 3.0, 101, x=8.0, y=1.0)
        _insert_timeseries(connection, 1, 4.0, 101, x=7.0, y=1.0)
        _insert_timeseries(connection, 1, 6.0, 101, x=5.0, y=1.0)

        _insert_fire(connection, 1, 4.0, 1)
        _insert_fire(connection, 1, 5.0, 1)
        _insert_fire(connection, 1, 5.0, 101)
        _insert_fire(connection, 1, 7.0, 2)
    return path


def test_adapter_requires_expected_schema_and_never_opens_writable(
    tmp_path: Path,
    rmuc_fixture: Path,
) -> None:
    adapter = RMUCSQLiteAdapter(rmuc_fixture)
    with adapter.connect() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE forbidden(value INT)")

    unsupported = tmp_path / "unsupported.sqlite"
    with _database(unsupported) as connection:
        connection.execute("CREATE TABLE matches(game_id INT)")
    with pytest.raises(RMUCSchemaError, match="missing tables"):
        RMUCSQLiteAdapter(unsupported)

    with pytest.raises(FileNotFoundError):
        RMUCSQLiteAdapter(tmp_path / "not-created.sqlite")
    assert not (tmp_path / "not-created.sqlite").exists()


def test_team_time_split_is_deterministic_team_disjoint_and_chronological(
    rmuc_fixture: Path,
) -> None:
    adapter = RMUCSQLiteAdapter(rmuc_fixture)
    first = adapter.build_team_time_split()
    second = adapter.build_team_time_split()

    assert first.assignments == second.assignments
    assert first.assignments == {1: "train", 2: "validation", 3: "test"}
    teams_by_split = {
        split: {
            team
            for component in first.components
            if component.split == split
            for team in component.teams
        }
        for split in ("train", "validation", "test")
    }
    assert teams_by_split["train"].isdisjoint(teams_by_split["validation"])
    assert teams_by_split["train"].isdisjoint(teams_by_split["test"])
    assert teams_by_split["validation"].isdisjoint(teams_by_split["test"])

    components = sorted(first.components, key=lambda component: component.first_start)
    assert components[0].last_start < components[1].first_start
    assert components[1].last_start < components[2].first_start


def test_sample_uses_masks_without_forward_fill_or_clipping(rmuc_fixture: Path) -> None:
    adapter = RMUCSQLiteAdapter(rmuc_fixture)
    config = TacticalWindowConfig(history_steps=3, future_horizon_s=2.0)
    sample = adapter.load_tactical_sample(TacticalSampleKey(1, 0, 4.0), config)

    assert sample.features.shape == (3, 16, len(RMUC_FEATURE_NAMES))
    assert sample.times_s.tolist() == [2.0, 3.0, 4.0]
    assert sample.robot_ids[:3].tolist() == [1, 2, 3]

    x = RMUC_FEATURE_NAMES.index("x")
    shots_17 = RMUC_FEATURE_NAMES.index("shots_17mm_cumulative")
    shots_42 = RMUC_FEATURE_NAMES.index("shots_42mm_cumulative")

    assert not sample.entity_present[1, 0]
    assert sample.features[1, 0].eq(0.0).all()
    assert not sample.feature_valid[1, 0].any()
    assert sample.features[2, 0, shots_17].item() == -5.0
    assert not sample.feature_valid[2, 0, shots_17]
    assert sample.features[0, 0, shots_42].item() == 0.0
    assert not sample.feature_valid[0, 0, shots_42]
    assert sample.features[2, 0, x].item() == 4.0

    assert sample.entity_present[2, 2]
    assert not sample.row_unique[2, 2]
    assert not sample.feature_valid[2, 2].any()
    assert not sample.position_valid[2, 1]  # Dynamic (0, 0) is a missing-position sentinel.
    assert not sample.feature_valid[2, 1, x]
    assert not sample.position_valid[2, 6]  # Building export uses the same sentinel.


def test_sample_derives_masked_future_intent_and_interval_fire(rmuc_fixture: Path) -> None:
    adapter = RMUCSQLiteAdapter(rmuc_fixture)
    sample = adapter.load_tactical_sample(
        TacticalSampleKey(1, 0, 4.0),
        TacticalWindowConfig(history_steps=3, future_horizon_s=2.0),
    )

    torch.testing.assert_close(sample.future_displacement_xy[0], torch.tensor([3.0, 4.0]))
    assert sample.future_displacement_valid[0]
    assert not sample.future_displacement_valid[1]  # Invalid anchor position.
    assert not sample.future_displacement_valid[2]
    assert not sample.future_displacement_valid[3]  # 9 m in 2 s exceeds 4 m/s.
    assert not sample.future_displacement_valid[4]  # Future x=29 is outside [0, 28].
    assert not sample.future_displacement_valid[5]  # Anchor HP is zero.
    assert not sample.future_displacement_valid[8]

    assert sample.fired[0]
    assert sample.fire_valid[0]
    assert not sample.fired[1]
    assert not sample.fire_valid[1]  # Engineer has no projectile-fire action.
    assert sample.fired[8]
    assert not sample.fire_valid[8]  # Opponent events are never own-team labels.


def test_blue_perspective_places_own_entities_first(rmuc_fixture: Path) -> None:
    adapter = RMUCSQLiteAdapter(rmuc_fixture)
    sample = adapter.load_tactical_sample(
        TacticalSampleKey(1, 1, 4.0),
        TacticalWindowConfig(history_steps=3, future_horizon_s=2.0),
    )

    assert sample.robot_ids[:8].tolist() == [101, 102, 103, 104, 106, 107, 110, 111]
    assert sample.own_entity[:8].all()
    assert not sample.own_entity[8:].any()
    assert sample.future_displacement_valid[0]


def test_dataset_indexes_only_the_requested_split(rmuc_fixture: Path) -> None:
    adapter = RMUCSQLiteAdapter(rmuc_fixture)
    manifest = adapter.build_team_time_split()
    config = TacticalWindowConfig(history_steps=3, future_horizon_s=2.0)
    dataset = RMUCTacticalDataset(adapter, manifest, "train", config)
    try:
        assert len(dataset) == 10
        assert {key.game_id for key in dataset.keys} == {1}
        samples = (dataset[2], dataset[3])
        assert samples[0].key.game_id == 1
        batch = collate_tactical_samples(samples)
        assert batch.features.shape == (2, 3, 16, len(RMUC_FEATURE_NAMES))
        assert batch.position_valid.shape == (2, 3, 16)
        assert batch.team.tolist() == [0, 1]
    finally:
        dataset.close()


def test_sample_rejects_history_or_future_outside_match(rmuc_fixture: Path) -> None:
    adapter = RMUCSQLiteAdapter(rmuc_fixture)
    config = TacticalWindowConfig(history_steps=3, future_horizon_s=2.0)

    with pytest.raises(ValueError, match="history"):
        adapter.load_tactical_sample(TacticalSampleKey(1, 0, 2.0), config)
    with pytest.raises(ValueError, match="future horizon"):
        adapter.load_tactical_sample(TacticalSampleKey(1, 0, 8.0), config)
