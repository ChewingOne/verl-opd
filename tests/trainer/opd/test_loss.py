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

import torch

from verl.trainer.opd.loss import build_opd_loss_mask, compute_opd_reverse_kl_loss


def test_shift_alignment_matches_next_token_prediction():
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    prompt_len = torch.tensor([1], dtype=torch.long)
    gen_len = torch.tensor([3], dtype=torch.long)

    logits = torch.zeros((1, 4, 5), dtype=torch.float32)
    logits[0, 0, 1] = 30.0
    logits[0, 1, 2] = 30.0
    logits[0, 2, 3] = 30.0

    out = compute_opd_reverse_kl_loss(
        student_logits=logits,
        teacher_output={"logits": logits.clone()},
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_len=prompt_len,
        gen_len=gen_len,
        mask_prompt_tokens=True,
        kl_reduction="mean",
    )

    assert torch.all(out["student_nll_token"] < 1e-5)
    assert torch.allclose(out["loss"], torch.zeros_like(out["loss"]), atol=1e-6)


def test_opd_mask_only_covers_completion_tokens_and_non_padding():
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    prompt_len = torch.tensor([3, 1], dtype=torch.long)
    gen_len = torch.tensor([2, 3], dtype=torch.long)

    mask = build_opd_loss_mask(
        attention_mask=attention_mask,
        prompt_len=prompt_len,
        gen_len=gen_len,
        mask_prompt_tokens=True,
    )

    expected = torch.tensor(
        [
            [0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(mask, expected)


def test_kl_is_zero_when_teacher_equals_student():
    torch.manual_seed(7)
    student_logits = torch.randn(2, 6, 11)
    input_ids = torch.randint(0, 11, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    prompt_len = torch.tensor([2, 3], dtype=torch.long)
    gen_len = torch.tensor([3, 2], dtype=torch.long)

    out = compute_opd_reverse_kl_loss(
        student_logits=student_logits,
        teacher_output={"logits": student_logits.clone()},
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_len=prompt_len,
        gen_len=gen_len,
        mask_prompt_tokens=True,
        kl_reduction="mean",
    )

    assert torch.allclose(out["loss"], torch.zeros_like(out["loss"]), atol=1e-6)


def test_kl_decreases_in_tiny_optimization_loop():
    torch.manual_seed(11)
    batch_size, seq_len, vocab = 3, 7, 13
    input_ids = torch.randint(0, vocab, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    prompt_len = torch.tensor([2, 3, 2], dtype=torch.long)
    gen_len = torch.tensor([3, 2, 4], dtype=torch.long)

    teacher_logits = torch.randn(batch_size, seq_len, vocab)
    student_logits = torch.nn.Parameter(torch.randn(batch_size, seq_len, vocab))
    optimizer = torch.optim.Adam([student_logits], lr=0.2)

    first_loss = None
    last_loss = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        out = compute_opd_reverse_kl_loss(
            student_logits=student_logits,
            teacher_output={"logits": teacher_logits},
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
            gen_len=gen_len,
            mask_prompt_tokens=True,
            kl_reduction="mean",
        )
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss

