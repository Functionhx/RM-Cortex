"""Clipped MAPPO update with a centralized critic."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from rm_train.buffer import RolloutBatch
from rm_train.config import MAPPOConfig
from rm_train.policy import SharedMAPPOPolicy


@dataclass(frozen=True)
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approximate_kl": self.approximate_kl,
            "clip_fraction": self.clip_fraction,
            "gradient_norm": self.gradient_norm,
        }


class MAPPO:
    def __init__(self, policy: SharedMAPPOPolicy, config: MAPPOConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1.0e-5)

    def update(self, rollout: RolloutBatch) -> UpdateMetrics:
        batch = rollout.flatten()
        advantages = batch.advantages
        batch.advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1.0e-8
        )
        totals = torch.zeros(6, device=batch.observations.device)
        updates = 0
        stop_early = False

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(batch.size, device=batch.observations.device)
            for start in range(0, batch.size, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                sample = batch.take(indices)
                evaluated = self.policy.evaluate_actions(
                    sample.observations,
                    sample.entities,
                    sample.entity_mask,
                    sample.central_state,
                    sample.target_mask,
                    sample.fire_mask,
                    sample.actions,
                    sample.agent_ids,
                )
                log_ratio = evaluated.log_prob - sample.old_log_prob
                ratio = log_ratio.exp()
                unclipped = ratio * sample.advantages
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    * sample.advantages
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                value_delta = evaluated.value - sample.old_value
                clipped_value = sample.old_value + value_delta.clamp(
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        torch.square(evaluated.value - sample.returns),
                        torch.square(clipped_value - sample.returns),
                    ).mean()
                )
                entropy = evaluated.entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio).to(torch.float32).mean()
                    )
                    totals += torch.stack(
                        (
                            policy_loss.detach(),
                            value_loss.detach(),
                            entropy.detach(),
                            approximate_kl.detach(),
                            clip_fraction.detach(),
                            torch.as_tensor(gradient_norm, device=totals.device),
                        )
                    )
                    updates += 1
                    if self.config.target_kl > 0 and approximate_kl > self.config.target_kl:
                        stop_early = True
                        break
            if stop_early:
                break

        if updates == 0:
            raise RuntimeError("MAPPO update did not receive a minibatch")
        means = (totals / updates).tolist()
        if not all(math.isfinite(value) for value in means):
            raise FloatingPointError("non-finite MAPPO update metrics")
        return UpdateMetrics(*means)
