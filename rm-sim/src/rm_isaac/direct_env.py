"""Import-safe facade for the optional Isaac Lab DirectMARLEnv."""

from __future__ import annotations

from dataclasses import dataclass

from rm_isaac.adapter import isaaclab_available


class IsaacLabUnavailableError(RuntimeError):
    """Raised when an Isaac-only entry point is used in a plain Torch process."""


if isaaclab_available():
    from rm_isaac.direct_env_runtime import (
        RMCortexDirectMARLEnv,
        RMCortexDirectMARLEnvCfg,
    )
else:

    @dataclass
    class RMCortexDirectMARLEnvCfg:  # type: ignore[no-redef]
        """Small placeholder that keeps configuration discovery import-safe."""

        num_envs: int = 1
        device: str = "cuda:0"
        seed: int = 0

    class RMCortexDirectMARLEnv:  # type: ignore[no-redef]
        """Unavailable-runtime sentinel; no Isaac module is imported."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise IsaacLabUnavailableError(
                "Isaac Lab is not importable. Launch this environment with the "
                "Isaac Lab Python entry point after installing Isaac Sim/Isaac Lab."
            )
