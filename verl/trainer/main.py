# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainer launcher that dispatches by ``algorithm.name``."""

import hydra

from verl.trainer.main_opd import run_opd
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


def _resolve_algorithm_name(config) -> str:
    algo_name = str(config.algorithm.get("name", "")).strip().lower()
    if algo_name:
        return algo_name

    # Backward-compatible fallback for existing PPO configs.
    if "adv_estimator" in config.algorithm:
        return "ppo"
    return ""


@hydra.main(config_path="config", config_name="opd_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    algo_name = _resolve_algorithm_name(config)

    if algo_name in {"opd", "on_policy_distill", "on_policy_distillation"}:
        run_opd(config)
        return
    if algo_name == "ppo":
        run_ppo(config)
        return
    raise ValueError(
        "Unsupported algorithm.name. "
        f"Got {algo_name!r}, expected one of ['ppo', 'opd', 'on_policy_distill', 'on_policy_distillation']."
    )


if __name__ == "__main__":
    main()
