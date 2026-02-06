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
"""Entry point for On-Policy Distillation (OPD) training."""

import hydra

from verl.trainer.opd.trainer import OPDTrainer
from verl.utils.device import auto_set_device


def run_opd(config) -> None:
    trainer = OPDTrainer(config=config)
    trainer.fit()


@hydra.main(config_path="config", config_name="opd_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_opd(config)


if __name__ == "__main__":
    main()
