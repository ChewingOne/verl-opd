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
"""Teacher worker for OPD that returns per-token logits/logprobs."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from verl.utils.torch_dtypes import PrecisionType


class TeacherLogProbWorker:
    """Frozen teacher inference worker for logit/logprob computation."""

    def __init__(
        self,
        *,
        model_path: str,
        dtype: str = "bf16",
        engine: str = "hf_transformers",
        device_map: str | None = "auto",
        approx_teacher_topk: int = 0,
        trust_remote_code: bool = False,
        device: torch.device | None = None,
    ) -> None:
        if engine != "hf_transformers":
            raise ValueError(f"Unsupported teacher_engine: {engine}")

        self.approx_teacher_topk = int(approx_teacher_topk)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = PrecisionType.to_dtype(dtype)

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if device_map not in (None, "null"):
            load_kwargs["device_map"] = device_map

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except Exception:
            load_kwargs.pop("device_map", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            self.model.to(self.device)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward teacher and return full logits or top-k logits."""
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        if self.approx_teacher_topk > 0:
            k = min(self.approx_teacher_topk, logits.size(-1))
            topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
            return {
                "topk_logits": topk_logits,
                "topk_indices": topk_indices,
            }
        return {"logits": logits}
