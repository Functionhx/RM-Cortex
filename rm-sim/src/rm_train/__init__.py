"""Pure-Torch MAPPO baseline and evaluation utilities."""

from rm_train.actions import decode_policy_actions, replace_team_actions
from rm_train.buffer import RolloutBatch, RolloutStorage
from rm_train.config import MAPPOConfig
from rm_train.evaluation import EvaluationReport, evaluate_against_scripted
from rm_train.mappo import MAPPO, UpdateMetrics
from rm_train.policy import PolicyAction, PolicyStep, SharedMAPPOPolicy
from rm_train.runner import MAPPOTrainingRunner, TrainingProgress

__all__ = [
    "EvaluationReport",
    "MAPPO",
    "MAPPOConfig",
    "MAPPOTrainingRunner",
    "PolicyAction",
    "PolicyStep",
    "RolloutBatch",
    "RolloutStorage",
    "SharedMAPPOPolicy",
    "TrainingProgress",
    "UpdateMetrics",
    "decode_policy_actions",
    "evaluate_against_scripted",
    "replace_team_actions",
]
