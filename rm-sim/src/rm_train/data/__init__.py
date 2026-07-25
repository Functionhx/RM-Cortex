"""Read-only adapters for offline competition data."""

from rm_train.data.policy_features import (
    AGENT_DIM,
    AGENT_FEATURE_NAMES,
    TacticalPolicyBatch,
    TacticalPolicyCollator,
    TacticalPolicyExample,
    collate_tactical_policy,
    featurize_tactical_sample,
)
from rm_train.data.rmuc_sqlite import (
    RMUC_FEATURE_NAMES,
    RMUC_ROBOT_IDS,
    RMUCSQLiteAdapter,
    RMUCSchemaError,
    RMUCTacticalDataset,
    SplitComponent,
    TacticalBatch,
    TacticalSample,
    TacticalSampleKey,
    TacticalWindowConfig,
    TeamTimeSplitManifest,
    collate_tactical_samples,
)

__all__ = [
    "AGENT_DIM",
    "AGENT_FEATURE_NAMES",
    "RMUC_FEATURE_NAMES",
    "RMUC_ROBOT_IDS",
    "RMUCSQLiteAdapter",
    "RMUCSchemaError",
    "RMUCTacticalDataset",
    "SplitComponent",
    "TacticalBatch",
    "TacticalPolicyBatch",
    "TacticalPolicyCollator",
    "TacticalPolicyExample",
    "TacticalSample",
    "TacticalSampleKey",
    "TacticalWindowConfig",
    "TeamTimeSplitManifest",
    "collate_tactical_policy",
    "collate_tactical_samples",
    "featurize_tactical_sample",
]
