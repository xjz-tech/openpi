from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    # 批量推理：接收多个 obs，统一做变换、拼 batch、一次前向，然后拆分并做输出变换
    def batch_infer(self, obs_list: list[dict]) -> list[dict]:  # type: ignore[misc]
        if not obs_list:
            return []

        # 输入变换逐个应用（与 infer 中一致）
        inputs_list = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in obs_list]

        # 拼成 batch（与 infer 中一致但不再添加新轴，而是直接 stack）
        batched_inputs = jax.tree.map(
            lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0), *inputs_list
        )

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        batched_outputs = {
            "state": batched_inputs["state"],
            "actions": self._sample_actions(
                sample_rng, _model.Observation.from_dict(batched_inputs), **self._sample_kwargs
            ),
        }

        # 拆分 batch 并做输出变换
        outputs_list: list[dict] = []
        num = batched_outputs["actions"].shape[0]
        model_time = time.monotonic() - start_time
        for i in range(num):
            single = jax.tree.map(lambda x: np.asarray(x[i, ...]), batched_outputs)
            single = self._output_transform(single)
            single["policy_timing"] = {"infer_ms": model_time * 1000}
            outputs_list.append(single)

        return outputs_list

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
