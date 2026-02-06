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
"""Trainer implementation for On-Policy Distillation (OPD)."""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler
from transformers import AutoModelForCausalLM

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn, get_dataset_class
from verl.utils.import_utils import load_extern_object
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_response_mask
from verl.utils.tracking import Tracking
from verl.workers.config import HFModelConfig
from verl.workers.ref.teacher_logprob_worker import TeacherLogProbWorker

from .loss import build_opd_loss_mask, compute_opd_reverse_kl_loss


@dataclass
class OPDRolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: torch.Tensor
    gen_len: torch.Tensor
    tokens: torch.Tensor


def _extract_step_from_ckpt_path(path: str) -> int | None:
    match = re.search(r"global_step_(\d+)", path)
    if match is None:
        return None
    return int(match.group(1))


def _create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    del is_train  # kept for API compatibility
    dataset_cls = get_dataset_class(data_config)
    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )


def _create_rl_sampler(data_config, dataset):
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        return curriculum_class(data_source=dataset, data_config=data_config)
    if data_config.shuffle:
        generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            generator.manual_seed(seed)
        return RandomSampler(data_source=dataset, generator=generator)
    return SequentialSampler(data_source=dataset)


class OPDTrainer:
    def __init__(self, config):
        self.config = config
        self.algorithm_cfg = config.algorithm
        self.optim_cfg = config.optim
        self.trainer_cfg = config.trainer
        self._setup_seed()
        self.device = self._resolve_device(self.trainer_cfg.get("device", "cuda"))

        self.model_cfg: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.tokenizer = self.model_cfg.tokenizer
        self.tokenizer.padding_side = "left"

        self.student_dtype = self._resolve_dtype(self.trainer_cfg.get("student_dtype", "bf16"))
        self.student = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.local_path,
            torch_dtype=self.student_dtype,
            trust_remote_code=self.model_cfg.trust_remote_code,
        ).to(self.device)
        self.student.train()

        teacher_model_path = self.algorithm_cfg.teacher_model_path
        if teacher_model_path in (None, "", "null"):
            teacher_model_path = self.model_cfg.path
        self.teacher_worker = TeacherLogProbWorker(
            model_path=teacher_model_path,
            dtype=self.algorithm_cfg.teacher_dtype,
            engine=self.algorithm_cfg.teacher_engine,
            device_map=self.algorithm_cfg.teacher_device_map,
            approx_teacher_topk=int(self.algorithm_cfg.approx_teacher_topk),
            trust_remote_code=self.model_cfg.trust_remote_code,
            device=self.device,
        )
        if int(self.algorithm_cfg.approx_teacher_topk) > 0:
            print(
                "WARNING: algorithm.approx_teacher_topk > 0 uses a top-k teacher approximation. "
                "This changes the KL objective and can introduce bias."
            )

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(self.optim_cfg.lr),
            betas=tuple(self.optim_cfg.betas),
            eps=float(self.optim_cfg.eps),
            weight_decay=float(self.optim_cfg.weight_decay),
        )

        self.max_grad_norm = float(self.optim_cfg.max_grad_norm)
        self.microbatch_size = max(int(self.algorithm_cfg.microbatch_size), 1)
        self.gradient_accumulation = max(int(self.algorithm_cfg.gradient_accumulation), 1)

        self.train_dataset = _create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.model_cfg.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.train_sampler = _create_rl_sampler(self.config.data, self.train_dataset)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=int(self.algorithm_cfg.batch_size_prompts),
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=int(self.config.data.get("dataloader_num_workers", 0)),
            drop_last=True,
            pin_memory=False,
        )

        self.total_training_steps = self._determine_total_training_steps()
        self.resume_global_step = self._load_checkpoint()
        self.global_step = self.resume_global_step
        self.optimizer_steps = 0

        self.tracking = Tracking(
            project_name=self.trainer_cfg.project_name,
            experiment_name=self.trainer_cfg.experiment_name,
            default_backend=self.trainer_cfg.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self._using_autocast = self.device.type == "cuda" and self.student_dtype in (torch.float16, torch.bfloat16)

    def _setup_seed(self) -> None:
        seed = int(self.config.trainer.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        return PrecisionType.to_dtype(dtype_name)

    def _determine_total_training_steps(self) -> int:
        if self.trainer_cfg.total_training_steps is not None:
            return int(self.trainer_cfg.total_training_steps)
        return int(len(self.train_dataloader) * int(self.trainer_cfg.total_epochs))

    def _resolve_resume_checkpoint(self) -> str | None:
        resume_mode = self.trainer_cfg.get("resume_mode", "auto")
        resume_from_path = self.trainer_cfg.get("resume_from_path", None)
        default_local_dir = self.trainer_cfg.default_local_dir
        tracker_file = os.path.join(default_local_dir, "latest_checkpointed_iteration.txt")

        if resume_mode == "disable":
            return None
        if resume_mode == "resume_path":
            return resume_from_path
        if resume_mode != "auto":
            raise ValueError(f"Unsupported resume_mode: {resume_mode}")
        if resume_from_path:
            return resume_from_path
        if not os.path.exists(tracker_file):
            return None
        with open(tracker_file, encoding="utf-8") as f:
            step = int(f.read().strip())
        ckpt_path = os.path.join(default_local_dir, f"global_step_{step}")
        return ckpt_path if os.path.isdir(ckpt_path) else None

    def _load_checkpoint(self) -> int:
        ckpt_dir = self._resolve_resume_checkpoint()
        if ckpt_dir is None:
            return 0

        ckpt_file = os.path.join(ckpt_dir, "opd_state.pt")
        if not os.path.exists(ckpt_file):
            return 0

        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        self.student.load_state_dict(state["student"])
        self.optimizer.load_state_dict(state["optimizer"])

        dataloader_state_path = os.path.join(ckpt_dir, "data_0.pt")
        if os.path.exists(dataloader_state_path):
            dataloader_state = torch.load(dataloader_state_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state)

        step = int(state.get("global_step") or _extract_step_from_ckpt_path(ckpt_dir) or 0)
        return step

    def _save_checkpoint(self, step: int) -> None:
        if self.trainer_cfg.save_freq <= 0:
            return

        os.makedirs(self.trainer_cfg.default_local_dir, exist_ok=True)
        ckpt_dir = os.path.join(self.trainer_cfg.default_local_dir, f"global_step_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {
            "student": self.student.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": step,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        torch.save(state, os.path.join(ckpt_dir, "opd_state.pt"))
        torch.save(self.train_dataloader.state_dict(), os.path.join(ckpt_dir, "data_0.pt"))

        tracker_file = os.path.join(self.trainer_cfg.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(tracker_file, "w", encoding="utf-8") as f:
            f.write(str(step))

        max_keep = self.trainer_cfg.get("max_ckpt_to_keep", None)
        if max_keep is not None:
            self._cleanup_old_checkpoints(max_keep=int(max_keep))

    def _cleanup_old_checkpoints(self, max_keep: int) -> None:
        if max_keep <= 0:
            return
        root = self.trainer_cfg.default_local_dir
        ckpt_dirs = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            step = _extract_step_from_ckpt_path(name)
            if os.path.isdir(path) and step is not None:
                ckpt_dirs.append((step, path))
        ckpt_dirs.sort(key=lambda x: x[0])
        while len(ckpt_dirs) > max_keep:
            _, old_path = ckpt_dirs.pop(0)
            for root_dir, dirs, files in os.walk(old_path, topdown=False):
                for file_name in files:
                    os.remove(os.path.join(root_dir, file_name))
                for dir_name in dirs:
                    os.rmdir(os.path.join(root_dir, dir_name))
            os.rmdir(old_path)

    def _prompt_to_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        return self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

    def _sample_rollout_batch(self, raw_prompts: np.ndarray) -> OPDRolloutBatch:
        prompt_texts = [self._prompt_to_text(item) for item in raw_prompts.tolist()]

        tokenized = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.config.data.max_prompt_length),
            add_special_tokens=False,
        )
        prompt_input_ids = tokenized["input_ids"].to(self.device)
        prompt_attention_mask = tokenized["attention_mask"].to(self.device)

        do_sample = float(self.algorithm_cfg.temperature) > 0.0
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.pad_token_id

        generation_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": int(self.algorithm_cfg.max_new_tokens),
            "num_return_sequences": int(self.algorithm_cfg.num_samples_per_prompt),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": float(self.algorithm_cfg.temperature),
                    "top_p": float(self.algorithm_cfg.top_p),
                    "top_k": int(self.algorithm_cfg.top_k),
                }
            )

        self.student.eval()
        with torch.no_grad():
            sequences = self.student.generate(**generation_kwargs)
        self.student.train()

        k = int(self.algorithm_cfg.num_samples_per_prompt)
        prompt_lens = prompt_attention_mask.sum(dim=-1)
        prompt_lens = prompt_lens.repeat_interleave(k, dim=0)
        repeated_prompt_ids = prompt_input_ids.repeat_interleave(k, dim=0)

        prompt_padded_len = prompt_input_ids.size(1)
        generated_tokens = sequences[:, prompt_padded_len:]
        if generated_tokens.size(1) < int(self.algorithm_cfg.max_new_tokens):
            pad_size = int(self.algorithm_cfg.max_new_tokens) - generated_tokens.size(1)
            pad_tokens = torch.full(
                (generated_tokens.size(0), pad_size),
                self.tokenizer.pad_token_id,
                dtype=generated_tokens.dtype,
                device=generated_tokens.device,
            )
            generated_tokens = torch.cat([generated_tokens, pad_tokens], dim=-1)

        response_mask = get_response_mask(
            generated_tokens,
            eos_token=[eos_token_id],
            dtype=torch.long,
        )
        gen_lens = response_mask.sum(dim=-1)

        packed_input_ids = []
        packed_attention_mask = []
        for idx in range(sequences.size(0)):
            plen = int(prompt_lens[idx].item())
            prompt_tokens = repeated_prompt_ids[idx, -plen:]
            completion_tokens = generated_tokens[idx]
            completion_mask = response_mask[idx]
            full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=-1)
            full_attention = torch.cat(
                [
                    torch.ones((plen,), dtype=torch.long, device=self.device),
                    completion_mask.to(torch.long),
                ],
                dim=-1,
            )
            packed_input_ids.append(full_tokens)
            packed_attention_mask.append(full_attention)

        max_seq_len = max(x.size(0) for x in packed_input_ids)
        batch_size = len(packed_input_ids)
        input_ids = torch.full(
            (batch_size, max_seq_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=self.device)
        for idx, (seq, mask) in enumerate(zip(packed_input_ids, packed_attention_mask, strict=True)):
            seq_len = seq.size(0)
            input_ids[idx, :seq_len] = seq
            attention_mask[idx, :seq_len] = mask

        return OPDRolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_len=prompt_lens.to(self.device),
            gen_len=gen_lens.to(self.device),
            tokens=input_ids,
        )

    def _forward_and_backward(self, batch: OPDRolloutBatch) -> dict[str, float]:
        total_kl_sum = 0.0
        total_student_nll_sum = 0.0
        total_teacher_nll_sum = 0.0
        total_loss_tokens = 0.0
        teacher_time = 0.0
        student_time = 0.0

        full_loss_mask = build_opd_loss_mask(
            attention_mask=batch.attention_mask,
            prompt_len=batch.prompt_len,
            gen_len=batch.gen_len,
            mask_prompt_tokens=bool(self.algorithm_cfg.mask_prompt_tokens),
        )
        global_denom = full_loss_mask.sum().clamp_min(1.0)

        for start in range(0, batch.input_ids.size(0), self.microbatch_size):
            end = min(start + self.microbatch_size, batch.input_ids.size(0))
            micro_input_ids = batch.input_ids[start:end]
            micro_attention_mask = batch.attention_mask[start:end]
            micro_prompt_len = batch.prompt_len[start:end]
            micro_gen_len = batch.gen_len[start:end]

            t0 = time.perf_counter()
            teacher_output = self.teacher_worker.forward_logits(
                input_ids=micro_input_ids,
                attention_mask=micro_attention_mask,
            )
            teacher_output = {k: v.to(self.device) for k, v in teacher_output.items()}
            teacher_time += time.perf_counter() - t0

            t1 = time.perf_counter()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.student_dtype,
                enabled=self._using_autocast,
            ):
                student_logits = self.student(
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                ).logits
            student_time += time.perf_counter() - t1

            loss_dict = compute_opd_reverse_kl_loss(
                student_logits=student_logits,
                teacher_output=teacher_output,
                input_ids=micro_input_ids,
                attention_mask=micro_attention_mask,
                prompt_len=micro_prompt_len,
                gen_len=micro_gen_len,
                mask_prompt_tokens=bool(self.algorithm_cfg.mask_prompt_tokens),
                kl_reduction=str(self.algorithm_cfg.kl_reduction),
                approx_teacher_topk=int(self.algorithm_cfg.approx_teacher_topk),
            )

            kl_token = loss_dict["kl_token"]
            loss_mask = loss_dict["loss_mask"]
            student_nll_token = loss_dict["student_nll_token"]
            teacher_nll_token = loss_dict["teacher_nll_token"]

            kl_sum = (kl_token * loss_mask).sum()
            if str(self.algorithm_cfg.kl_reduction) == "sum":
                loss_for_backward = kl_sum
            else:
                loss_for_backward = kl_sum / global_denom

            (loss_for_backward / self.gradient_accumulation).backward()

            with torch.no_grad():
                mask_sum = loss_mask.sum().item()
                total_loss_tokens += mask_sum
                total_kl_sum += kl_sum.item()
                total_student_nll_sum += (student_nll_token * loss_mask).sum().item()
                total_teacher_nll_sum += (teacher_nll_token * loss_mask).sum().item()

        denom = max(total_loss_tokens, 1.0)
        metrics = {
            "opd/kl_mean": total_kl_sum / denom,
            "opd/nll_student": total_student_nll_sum / denom,
            "opd/nll_teacher_on_student_samples": total_teacher_nll_sum / denom,
            "opd/tokens": total_loss_tokens,
            "perf/opd_teacher_forward_time": teacher_time,
            "perf/opd_student_forward_time": student_time,
        }
        return metrics

    def fit(self) -> None:
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)

        step_in_accum = 0
        start_epoch = self.global_step // max(len(self.train_dataloader), 1)
        log_every = max(int(self.algorithm_cfg.log_every), 1)

        for epoch in range(start_epoch, int(self.trainer_cfg.total_epochs)):
            if self.global_step >= self.total_training_steps:
                break
            for batch_dict in self.train_dataloader:
                if self.global_step >= self.total_training_steps:
                    break

                self.global_step += 1
                step_in_accum += 1
                step_start = time.perf_counter()

                rollout_batch = self._sample_rollout_batch(batch_dict["raw_prompt"])
                metrics = self._forward_and_backward(rollout_batch)

                do_step = (step_in_accum % self.gradient_accumulation == 0) or (
                    self.global_step >= self.total_training_steps
                )
                if do_step:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.optimizer_steps += 1

                elapsed = max(time.perf_counter() - step_start, 1e-8)
                metrics["perf/tokens_per_sec"] = metrics["opd/tokens"] / elapsed
                metrics["training/global_step"] = self.global_step
                metrics["training/epoch"] = epoch
                metrics["training/optimizer_step"] = self.optimizer_steps

                if self.global_step == 1 or self.global_step % log_every == 0:
                    self.tracking.log(data=metrics, step=self.global_step)

                if self.trainer_cfg.save_freq > 0 and (
                    self.global_step % int(self.trainer_cfg.save_freq) == 0
                    or self.global_step >= self.total_training_steps
                ):
                    self._save_checkpoint(step=self.global_step)
