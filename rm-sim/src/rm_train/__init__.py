"""Pure-Torch MAPPO baseline and evaluation utilities."""

from rm_train.actions import decode_policy_actions, replace_team_actions
from rm_train.baseline_evaluation import (
    ControllerEvaluationReport,
    ControllerLegReport,
    evaluate_world_controllers,
)
from rm_train.buffer import RolloutBatch, RolloutStorage
from rm_train.config import MAPPOConfig
from rm_train.evaluation import EvaluationReport, evaluate_against_scripted
from rm_train.imitation import TacticalBehaviorCloner, load_imitation_actor_checkpoint
from rm_train.imitation_runner import ImitationConfig, ImitationTrainingRunner
from rm_train.mappo import MAPPO, UpdateMetrics
from rm_train.policy import PolicyAction, PolicyStep, SharedMAPPOPolicy
from rm_train.runner import MAPPOTrainingRunner, TrainingProgress

__all__ = [
    "EvaluationReport",
    "ControllerEvaluationReport",
    "ControllerLegReport",
    "ImitationConfig",
    "ImitationTrainingRunner",
    "MAPPO",
    "MAPPOConfig",
    "MAPPOTrainingRunner",
    "PolicyAction",
    "PolicyStep",
    "RolloutBatch",
    "RolloutStorage",
    "SharedMAPPOPolicy",
    "TacticalBehaviorCloner",
    "TrainingProgress",
    "UpdateMetrics",
    "decode_policy_actions",
    "evaluate_against_scripted",
    "evaluate_world_controllers",
    "load_imitation_actor_checkpoint",
    "replace_team_actions",
]
