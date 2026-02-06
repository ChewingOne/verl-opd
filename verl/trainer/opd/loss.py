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
"""Loss utilities for On-Policy Distillation (OPD)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_opd_loss_mask(
    attention_mask: torch.Tensor,
    prompt_len: torch.Tensor,
    gen_len: torch.Tensor,
    *,
    mask_prompt_tokens: bool = True,
) -> torch.Tensor:
    """Build loss mask aligned to shifted logits with shape [B, T-1]."""
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be [B, T], got {attention_mask.shape}")

    batch_size, seq_len = attention_mask.shape
    if seq_len < 2:
        return torch.zeros((batch_size, 0), dtype=torch.float32, device=attention_mask.device)

    valid_next_token_mask = attention_mask[:, 1:].to(torch.bool)
    if not mask_prompt_tokens:
        return valid_next_token_mask.to(torch.float32)

    pos = torch.arange(seq_len - 1, device=attention_mask.device).unsqueeze(0).expand(batch_size, -1)
    start = torch.clamp(prompt_len - 1, min=0).unsqueeze(1)
    end = (prompt_len - 1 + gen_len).unsqueeze(1)
    generation_mask = (pos >= start) & (pos < end)
    return (generation_mask & valid_next_token_mask).to(torch.float32)


def _get_teacher_topk_token_logp(
    next_token_ids: torch.Tensor,
    topk_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    vocab_size: int,
    tail_eps: float,
) -> torch.Tensor:
    """Approximate teacher token log-prob from top-k teacher output."""
    k = topk_logits.size(-1)
    if k >= vocab_size:
        full_logits = torch.full(
            (*topk_logits.shape[:-1], vocab_size),
            fill_value=torch.finfo(topk_logits.dtype).min,
            dtype=topk_logits.dtype,
            device=topk_logits.device,
        )
        full_logits.scatter_(dim=-1, index=topk_indices, src=topk_logits)
        full_logp = F.log_softmax(full_logits, dim=-1)
        gathered = torch.gather(full_logp, dim=-1, index=next_token_ids.unsqueeze(-1))
        return gathered.squeeze(-1)

    if tail_eps <= 0.0:
        tail_eps = 1e-8

    topk_logp = F.log_softmax(topk_logits, dim=-1) + math.log(max(1.0 - tail_eps, 1e-8))
    in_topk = topk_indices.eq(next_token_ids.unsqueeze(-1))
    in_topk_any = in_topk.any(dim=-1)

    in_topk_logp = (topk_logp * in_topk.to(topk_logp.dtype)).sum(dim=-1)
    tail_logp = math.log(tail_eps / max(vocab_size - k, 1))

    return torch.where(in_topk_any, in_topk_logp, torch.full_like(in_topk_logp, tail_logp))


def _reverse_kl_with_topk_teacher(
    logp_s: torch.Tensor,
    p_s: torch.Tensor,
    topk_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    vocab_size: int,
    tail_eps: float,
) -> torch.Tensor:
    """Approximate reverse KL using teacher top-k logits with a uniform tail."""
    if tail_eps <= 0.0:
        tail_eps = 1e-8

    k = topk_logits.size(-1)
    if k >= vocab_size:
        full_logits = torch.full(
            (*topk_logits.shape[:-1], vocab_size),
            fill_value=torch.finfo(topk_logits.dtype).min,
            dtype=topk_logits.dtype,
            device=topk_logits.device,
        )
        full_logits.scatter_(dim=-1, index=topk_indices, src=topk_logits)
        logp_t = F.log_softmax(full_logits, dim=-1)
        return torch.sum(p_s * (logp_s - logp_t), dim=-1)

    topk_logp_t = F.log_softmax(topk_logits, dim=-1) + math.log(max(1.0 - tail_eps, 1e-8))
    tail_logp_t = math.log(tail_eps / max(vocab_size - k, 1))

    p_s_topk = torch.gather(p_s, dim=-1, index=topk_indices)
    logp_s_topk = torch.gather(logp_s, dim=-1, index=topk_indices)

    topk_term = torch.sum(p_s_topk * (logp_s_topk - topk_logp_t), dim=-1)

    sum_all_ps_logps = torch.sum(p_s * logp_s, dim=-1)
    sum_topk_ps_logps = torch.sum(p_s_topk * logp_s_topk, dim=-1)
    sum_tail_ps_logps = sum_all_ps_logps - sum_topk_ps_logps
    tail_mass = 1.0 - torch.sum(p_s_topk, dim=-1)
    tail_term = sum_tail_ps_logps - tail_mass * tail_logp_t

    return topk_term + tail_term


def compute_opd_reverse_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_output: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: torch.Tensor,
    gen_len: torch.Tensor,
    mask_prompt_tokens: bool = True,
    kl_reduction: str = "mean",
    approx_teacher_topk: int = 0,
    approx_teacher_tail_eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute token-level reverse KL KL(pi_student || pi_teacher)."""
    if student_logits.dim() != 3:
        raise ValueError(f"student_logits must be [B, T, V], got {student_logits.shape}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be [B, T], got {input_ids.shape}")

    logp_s = F.log_softmax(student_logits[:, :-1, :], dim=-1)
    p_s = torch.exp(logp_s)

    next_token_ids = input_ids[:, 1:]
    student_next_token_logp = torch.gather(logp_s, dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)
    student_nll_token = -student_next_token_logp

    vocab_size = student_logits.size(-1)
    if approx_teacher_topk > 0:
        topk_logits = teacher_output["topk_logits"][:, :-1, :]
        topk_indices = teacher_output["topk_indices"][:, :-1, :]
        kl_token = _reverse_kl_with_topk_teacher(
            logp_s=logp_s,
            p_s=p_s,
            topk_logits=topk_logits,
            topk_indices=topk_indices,
            vocab_size=vocab_size,
            tail_eps=approx_teacher_tail_eps,
        )
        teacher_next_token_logp = _get_teacher_topk_token_logp(
            next_token_ids=next_token_ids,
            topk_logits=topk_logits,
            topk_indices=topk_indices,
            vocab_size=vocab_size,
            tail_eps=approx_teacher_tail_eps,
        )
    else:
        teacher_logits = teacher_output["logits"]
        logp_t = F.log_softmax(teacher_logits[:, :-1, :], dim=-1)
        kl_token = torch.sum(p_s * (logp_s - logp_t), dim=-1)
        teacher_next_token_logp = torch.gather(logp_t, dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)

    teacher_nll_token = -teacher_next_token_logp
    loss_mask = build_opd_loss_mask(
        attention_mask=attention_mask,
        prompt_len=prompt_len,
        gen_len=gen_len,
        mask_prompt_tokens=mask_prompt_tokens,
    )
    weighted_kl = kl_token * loss_mask
    if kl_reduction == "sum":
        loss = weighted_kl.sum()
    elif kl_reduction == "mean":
        denom = loss_mask.sum().clamp_min(1.0)
        loss = weighted_kl.sum() / denom
    else:
        raise ValueError(f"Unsupported kl_reduction: {kl_reduction}")

    return {
        "loss": loss,
        "kl_token": kl_token,
        "loss_mask": loss_mask,
        "student_nll_token": student_nll_token,
        "teacher_nll_token": teacher_nll_token,
    }
