"""Training orchestration for the pure-Torch RM-Cortex environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from rm_train.actions import decode_policy_actions
from rm_train.buffer import RolloutStorage
from rm_train.config import MAPPOConfig
from rm_train.mappo import MAPPO, UpdateMetrics
from rm_train.policy import SharedMAPPOPolicy
from rm_world import TorchEnvConfig, TorchRMArena, WorldObservation


@dataclass(frozen=True)
class TrainingProgress:
    update: int
    environment_steps: int
    mean_step_reward: float
    metrics: UpdateMetrics

    def as_dict(self) -> dict[str, float | int]:
        return {
            "update": self.update,
            "environment_steps": self.environment_steps,
            "mean_step_reward": self.mean_step_reward,
            **self.metrics.as_dict(),
        }


class MAPPOTrainingRunner:
    def __init__(
        self,
        config: MAPPOConfig,
        *,
        policy: SharedMAPPOPolicy | None = None,
        environment: TorchRMArena | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = config.resolved_device
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(config.seed)
        self.environment = environment or TorchRMArena(
            TorchEnvConfig(
                num_envs=config.num_envs,
                device=self.device,
                seed=config.seed,
                validate_referee=False,
            )
        )
        self.policy = policy or SharedMAPPOPolicy(
            hidden_dim=config.hidden_dim,
            role_embedding_dim=config.role_embedding_dim,
        )
        self.policy.to(self.device)
        self.algorithm = MAPPO(self.policy, config)
        self.observation = self.environment.reset(seed=config.seed)
        self.environment_steps = 0
        self.update_index = 0

    def _next_value(self, observation: WorldObservation) -> torch.Tensor:
        agent_ids = self.policy.default_agent_ids(
            observation.agents.shape[:-1],
            observation.agents.device,
        )
        return self.policy.value(observation.central, agent_ids)

    def collect_rollout(self) -> tuple[RolloutStorage, float]:
        storage = RolloutStorage()
        reward_total = torch.zeros((), device=self.environment.game.device)
        for _ in range(self.config.rollout_steps):
            current = self.observation
            with torch.no_grad():
                policy_step = self.policy.act(
                    current.agents,
                    current.central,
                    current.target_mask,
                    current.fire_mask,
                )
            world_action = decode_policy_actions(
                self.environment.game,
                self.environment.world,
                policy_step.action,
            )
            result = self.environment.step(world_action)
            storage.add(
                observations=current.agents,
                central_state=current.central,
                target_mask=current.target_mask,
                fire_mask=current.fire_mask,
                actions=policy_step.action,
                log_prob=policy_step.log_prob,
                value=policy_step.value,
                reward=result.reward,
                done=result.terminated,
            )
            reward_total += result.reward.mean()
            self.observation = result.observation
            done_ids = torch.nonzero(result.terminated, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                self.observation = self.environment.reset(done_ids)
        self.environment_steps += self.config.num_envs * self.config.rollout_steps
        return storage, float(reward_total.item() / self.config.rollout_steps)

    def train_update(self) -> TrainingProgress:
        self.policy.eval()
        storage, mean_reward = self.collect_rollout()
        with torch.no_grad():
            next_value = self._next_value(self.observation)
        rollout = storage.finish(
            next_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        self.policy.train()
        metrics = self.algorithm.update(rollout)
        self.update_index += 1
        return TrainingProgress(
            update=self.update_index,
            environment_steps=self.environment_steps,
            mean_step_reward=mean_reward,
            metrics=metrics,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.algorithm.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "update": self.update_index,
                "environment_steps": self.environment_steps,
            },
            destination,
        )
        return destination

    def train(self) -> tuple[list[TrainingProgress], Path]:
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(self.config.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")

        progress: list[TrainingProgress] = []
        metrics_path = output / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as stream:
            for _ in range(self.config.total_updates):
                current = self.train_update()
                progress.append(current)
                stream.write(json.dumps(current.as_dict(), sort_keys=True) + "\n")
                stream.flush()
                if (
                    self.config.checkpoint_interval > 0
                    and current.update % self.config.checkpoint_interval == 0
                ):
                    self.save_checkpoint(output / f"checkpoint_{current.update:05d}.pt")
        final_checkpoint = self.save_checkpoint(output / "latest.pt")
        return progress, final_checkpoint
