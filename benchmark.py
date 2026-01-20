#!/usr/bin/env python3



# =========================
# 0) CLI
# =========================

import argparse


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--target",
        choices=["noflash", "wedlm"],
        required=True,
        help="noflash = HF AR/MDLM/BD3LM suite; wedlm = Tencent WeDLM only",
    )
    p.add_argument("--prompts", default="prompts.jsonl")
    p.add_argument("--out", default="results.jsonl")
    p.add_argument("--append", action="store_true", help="Append to out file instead of overwriting")

    # Cross-env DAG plumbing
    p.add_argument(
        "--dag_mode",
        choices=["make_buffer", "consume_buffer"],
        default=None,
        help="For cross-env DAG. make_buffer writes *_buf.jsonl; consume_buffer reads it and writes results.",
    )
    p.add_argument("--buf", default=None, help="Path to JSONL buffer file for cross-env DAG (read or write).")
    p.add_argument("--dag_dir", default="dag_buffers", help="Directory to store/read DAG buffer files.")
    p.add_argument("--dag_tag", default=None, help="Optional override model_tag for DAG outputs (else auto).")
    p.add_argument("--limit", type=int, default=None, help="Optional limit number of prompts processed.")

    # In-env DAG between AR<->BD3LM is always available in noflash mode (no extra flags needed).
    return p.parse_args()


# =========================
# 1) Imports / Constants
# =========================

import copy
import json
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np  # (kept if you later expand metrics)
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

MODELS = {
    "ar": "Qwen/Qwen3-0.6B",
    "mdlm": "dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1",
    "bd3lm": "dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1",
    "ar_8b": "Qwen/Qwen3-8B-Base",
    "diff8": "tencent/WeDLM-8B-Base",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

END_MARK = "<END>"


# =========================
# 2) JSONL / IO utilities
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_prompts(path: str, limit: Optional[int] = None) -> List[dict]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            prompts.append(json.loads(line))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


# =========================
# 3) Prompt formatting
# =========================

STRUCT_SUFFIX = (
    "\n\nAnswer using exactly these headings:\n"
    "Goal:\nControls:\nReadout:\nAnalysis:\nPitfalls:\n"
    "- Goal: 1–2 sentences.\n"
    "- Controls/Readout/Analysis/Pitfalls: 2–4 bullets each.\n"
    "- Every bullet must be concrete (mention an assay, control type, comparison, or statistic).\n"
    "- No repeated bullet stems (don’t start multiple bullets with the same 2–3 words).\n"
    "Keep total length under 250 tokens.\n"
    f"End your answer with {END_MARK} on its own line."
)

Z0_SUFFIX = (
    "\n\nKeep total length under 250 tokens.\n"
    f"End your answer with {END_MARK} on its own line."
)


def apply_prompt_style(prompt: str, style: str) -> str:
    if style == "z0":
        return prompt + Z0_SUFFIX
    if style == "s1":
        return prompt + STRUCT_SUFFIX
    raise ValueError(f"unknown style: {style}")


def format_prompt(tok, user_text: str, use_format: bool = True) -> str:
    if use_format and hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return user_text


def format_prompt_wedlm(tok, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =========================
# 4) Model loaders
# =========================

def load_ar(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_mlm(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


# =========================
# 5) Shared diffusion helpers
# =========================

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    rem = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
    for i in range(mask_num.size(0)):
        out[i, : rem[i]] += 1
    return out


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    g = (-torch.log(noise)) ** temperature
    return logits.exp() / g


# =========================
# 6) Runners: AR / MDLM / BD3LM
# =========================

@torch.no_grad()
def run_ar(tok, model, prompt: str, max_new_tokens=256, temperature=0.6, top_p=0.95):
    prompt_fmt = format_prompt(tok, prompt)
    inputs = tok(prompt_fmt, return_tensors="pt").to(model.device)

    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    dt = time.time() - t0

    gen_ids = out[0, inputs["input_ids"].shape[-1] :]
    decoded = tok.decode(gen_ids, skip_special_tokens=True)

    # Strip chain-of-thought if present
    if "</think>" in decoded:
        answer = decoded.split("</think>", 1)[1].strip()
    else:
        answer = decoded.strip()

    in_tokens = int(inputs["input_ids"].shape[-1])
    out_tokens = int(gen_ids.shape[-1])
    return answer, dt, in_tokens, out_tokens


@torch.no_grad()
def run_mdlm_quickstart(
    tok,
    model,
    prompt: str,
    max_new_tokens=256,
    steps=256,
    block_size=8,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
):
    device = model.device

    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id or tok.eos_token_id or tok.mask_token_id
    mask_id = tok.mask_token_id

    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt},
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    prompt_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_lens = torch.tensor([prompt_ids.shape[1]], dtype=torch.long, device=device)

    total_length = int(prompt_lens.max().item() + max_new_tokens)
    x = torch.full((1, total_length), pad_id, dtype=torch.long, device=device)
    L = int(prompt_lens.item())
    x[0, :L] = prompt_ids[0, :L]
    x[0, L : L + max_new_tokens] = mask_id

    prompt_index = torch.arange(total_length, device=device).unsqueeze(0) < prompt_lens.unsqueeze(1)
    positions = torch.arange(total_length, device=device)

    assert max_new_tokens % block_size == 0, "max_new_tokens must be divisible by block_size for MDLM"
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks for MDLM"
    steps_per_block = steps // num_blocks

    t0 = time.time()

    for b in range(num_blocks):
        block_start = prompt_lens + b * block_size
        block_end = block_start + block_size

        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        num_transfer = get_num_transfer_tokens(init_block_mask, steps_per_block)

        for t in range(steps_per_block):
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_noisy = add_gumbel_noise(logits, temperature)
            x0 = logits_noisy.argmax(dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                conf = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                conf = torch.rand_like(x0, dtype=torch.float)
            else:
                raise ValueError(remasking)

            conf = torch.where(block_mask, conf, torch.full_like(conf, -float("inf")))
            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            k = int(num_transfer[0, t].item())
            if k > 0:
                _, idx = torch.topk(conf[0], k=k)
                transfer_index[0, idx] = True

            x[transfer_index] = x0[transfer_index]

    dt = time.time() - t0

    new_ids = x[0, L : L + max_new_tokens].detach().cpu().tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)

    in_tokens = int(L)
    out_tokens = int(max_new_tokens)
    return text, dt, in_tokens, out_tokens


# ---- BD3LM core (no auto-continue helpers) ----

def build_staircase_attention_mask(x: torch.Tensor, block_size: int, pad_id: int):
    B, T = x.shape
    device = x.device

    valid = x != pad_id
    pos_raw = torch.cumsum(valid.long(), dim=-1)
    position_ids = torch.where(valid, pos_raw - 1, torch.zeros_like(pos_raw)).long()

    col = torch.arange(T, device=device)
    block_ids = (col // block_size).view(1, T).expand(B, T)
    block_ids = torch.where(valid, block_ids, torch.full_like(block_ids, -1))

    q = block_ids.view(B, 1, T, 1)
    k = block_ids.view(B, 1, 1, T)
    attn = (k <= q) & (q >= 0) & (k >= 0)

    return attn, position_ids


def diffusion_step_block(
    logits: torch.Tensor,
    x_block: torch.Tensor,
    mask_block: torch.Tensor,
    num_transfer: torch.Tensor,
    temperature: float,
    remasking: str,
):
    B, L, _ = logits.shape
    if not mask_block.any():
        return x_block

    noisy = add_gumbel_noise(logits, temperature)
    x0 = noisy.argmax(dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        conf = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        conf = torch.rand((B, L), device=logits.device)
    else:
        raise ValueError(remasking)

    x0 = torch.where(mask_block, x0, x_block)

    neg_inf = torch.full_like(conf, -float("inf"))
    conf = torch.where(mask_block, conf, neg_inf)

    commit = torch.zeros_like(x_block, dtype=torch.bool)
    for i in range(B):
        k = int(num_transfer[i].item())
        if k > 0:
            valid_k = (conf[i] > -float("inf")).sum().item()
            k = min(k, valid_k)
            if k > 0:
                _, idx = torch.topk(conf[i], k)
                commit[i, idx] = True

    out = x_block.clone()
    out[commit] = x0[commit]
    return out


@torch.no_grad()
def bd3lm_generate(
    model,
    tokenizer,
    prompt: torch.Tensor,
    steps=128,
    max_new_tokens=128,
    block_size=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
):
    device = model.device
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.mask_token_id

    x = prompt.to(device).long()
    if x.dim() == 1:
        x = x.unsqueeze(0)

    B = x.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = math.ceil(steps / num_blocks)
    generated = 0

    while generated < max_new_tokens:
        if finished.all():
            break

        T_prefix = x.size(1)
        offset = T_prefix % block_size
        room = block_size if offset == 0 else block_size - offset
        cur_len = min(room, max_new_tokens - generated)
        if cur_len <= 0:
            break

        attn_pfx, pos_pfx = build_staircase_attention_mask(x, block_size, pad_id)
        out = model(x, attention_mask=attn_pfx, position_ids=pos_pfx, use_cache=True)
        cond_past = out.past_key_values

        if cfg_scale > 0:
            un_x = x.clone()
            un_x[:] = mask_id
            out_un = model(un_x, attention_mask=attn_pfx, position_ids=pos_pfx, use_cache=True)
            uncond_past = out_un.past_key_values
        else:
            uncond_past = None

        block = torch.full((B, cur_len), mask_id, device=device, dtype=torch.long)
        block[finished] = pad_id
        x = torch.cat([x, block], dim=1)
        T_total = x.size(1)

        block_mask = x[:, -cur_len:] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        eff_steps = num_transfer.size(1)

        full_attn, full_pos = build_staircase_attention_mask(x, block_size, pad_id)
        attn_blk = full_attn[:, :, T_prefix:T_total, :]
        pos_blk = full_pos[:, T_prefix:T_total]

        for t in range(eff_steps):
            x_blk = x[:, T_prefix:T_total]
            m_blk = x_blk == mask_id

            cond_logits = model(
                x_blk,
                attention_mask=attn_blk,
                position_ids=pos_blk,
                past_key_values=copy.deepcopy(cond_past),
                use_cache=False,
            ).logits

            logits = cond_logits
            if cfg_scale > 0:
                un_logits = model(
                    x_blk,
                    attention_mask=attn_blk,
                    position_ids=pos_blk,
                    past_key_values=copy.deepcopy(uncond_past),
                    use_cache=False,
                ).logits
                logits = un_logits + (cfg_scale + 1.0) * (cond_logits - un_logits)

            x_blk_new = diffusion_step_block(
                logits, x_blk, m_blk, num_transfer[:, t], temperature, remasking
            )
            x[:, T_prefix:T_total] = x_blk_new

            if tokenizer.eos_token_id is not None:
                finished |= (x_blk_new == tokenizer.eos_token_id).any(dim=1)
            if finished.all():
                break

        generated += cur_len
        if finished.all():
            break

    return x


@torch.no_grad()
def run_bd3lm_quickstart(
    tok,
    model,
    prompt: str,
    max_new_tokens=256,
    steps=256,
    block_size=16,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
):
    device = model.device

    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt},
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    x0 = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    L0 = int(x0.shape[1])

    t0 = time.time()
    x = bd3lm_generate(
        model=model,
        tokenizer=tok,
        prompt=x0,
        steps=steps,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
    )
    dt = time.time() - t0

    gen_ids = x[0, L0:].detach().cpu().tolist()
    text = tok.decode(gen_ids, skip_special_tokens=True)

    out_tokens = int(x.shape[1] - L0)
    return text, dt, L0, out_tokens


# =========================
# 7) WeDLM runner
# =========================

def load_wedlm_engine(model_id: str):
    from wedlm import LLM
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id)
    return tok, llm


@torch.no_grad()
def run_wedlm(tok, llm, prompt: str, max_new_tokens=256, temperature=0.0):
    from wedlm import SamplingParams as SParams

    text_in = format_prompt_wedlm(tok, prompt)

    t0 = time.time()
    outputs = llm.generate([text_in], SParams(temperature=temperature, max_tokens=max_new_tokens))
    dt = time.time() - t0

    text_out = outputs[0]["text"].strip()

    in_tokens = int(tok(text_in, return_tensors="pt")["input_ids"].shape[-1])
    out_tokens = int(tok(text_out, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[-1])

    if "</think>" in text_out:
        text_out = text_out.split("</think>", 1)[1].strip()

    return text_out, dt, in_tokens, out_tokens


# =========================
# 8) Agent utilities (draft -> critique -> revise)
# =========================

def strip_after_end(text: str) -> str:
    if not text:
        return text
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(rf"(?m)^\s*{re.escape(END_MARK)}\s*$", t)
    if not m:
        return t.strip()
    return t[: m.end()].strip()


def ensure_end_line(text: str) -> str:
    t = (text or "").rstrip()
    trimmed = strip_after_end(t)
    if re.search(rf"(?m)^\s*{re.escape(END_MARK)}\s*$", trimmed):
        return trimmed
    return trimmed + "\n" + END_MARK


CRITIQUE_SUFFIX = (
    "\n\nYou are a strict reviewer. Critique the draft for:\n"
    "1) missing/incorrect required headings or formatting\n"
    "2) factual/biomedical plausibility issues\n"
    "3) contradictions or unclear steps\n"
    "4) verbosity (must stay <250 tokens)\n\n"
    "Return exactly 3 bullet points, each starting with '- '.\n"
    "Do NOT rewrite the answer.\n"
    f"End with {END_MARK} on its own line."
)

REVISION_SUFFIX = (
    "\n\nRewrite the draft into a final answer that fixes the critique.\n"
    "Rules:\n"
    "- Preserve the required headings and formatting implied by the task prompt.\n"
    "- Controls/Readout/Analysis/Pitfalls: 2–4 bullets each.\n"
    "- Total length under 250 tokens.\n"
    f"- End with {END_MARK} on its own line."
)


def critique_prompt(user_prompt: str, draft: str) -> str:
    return f"Task prompt:\n{user_prompt}\n\nDraft answer:\n{draft}\n" + CRITIQUE_SUFFIX


def revision_prompt(user_prompt: str, draft: str, critique: str) -> str:
    return (
        f"Task prompt:\n{user_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n"
        + REVISION_SUFFIX
    )


def run_agent_loop(
    runner_fn,
    tok,
    model,
    user_prompt: str,
    gen_cfg: dict,
    critique_cfg: dict,
    revise_cfg: dict,
):
    draft, dt1, in1, out1 = runner_fn(tok, model, user_prompt, **gen_cfg)
    draft = ensure_end_line(draft)

    cprompt = critique_prompt(user_prompt, strip_after_end(draft))
    critique, dt2, in2, out2 = runner_fn(tok, model, cprompt, **critique_cfg)
    critique = ensure_end_line(critique)

    rprompt = revision_prompt(user_prompt, strip_after_end(draft), strip_after_end(critique))
    revised, dt3, in3, out3 = runner_fn(tok, model, rprompt, **revise_cfg)
    revised = ensure_end_line(revised)

    meta = {
        "draft": {"latency_s": dt1, "input_tokens": in1, "output_tokens": out1, "cfg": gen_cfg},
        "critique": {"latency_s": dt2, "input_tokens": in2, "output_tokens": out2, "cfg": critique_cfg},
        "revise": {"latency_s": dt3, "input_tokens": in3, "output_tokens": out3, "cfg": revise_cfg},
        "total_latency_s": (dt1 or 0) + (dt2 or 0) + (dt3 or 0),
        "total_output_tokens": (out1 or 0) + (out2 or 0) + (out3 or 0),
    }
    return revised, meta, draft, critique


def run_ar_agent_loop(
    tok,
    model,
    user_prompt: str,
    gen_cfg: dict,
    critique_cfg: dict = None,
    revise_cfg: dict = None,
):
    if critique_cfg is None:
        critique_cfg = {**gen_cfg, "temperature": 0.1, "top_p": 1.0, "max_new_tokens": 128}
    if revise_cfg is None:
        revise_cfg = {
            **gen_cfg,
            "temperature": 0.2,
            "top_p": 1.0,
            "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
        }

    draft, dt1, in1, out1 = run_ar(tok, model, user_prompt, **gen_cfg)
    draft = ensure_end_line(draft)

    cprompt = critique_prompt(user_prompt, strip_after_end(draft))
    critique, dt2, in2, out2 = run_ar(tok, model, cprompt, **critique_cfg)
    critique = ensure_end_line(critique)

    rprompt = revision_prompt(user_prompt, strip_after_end(draft), strip_after_end(critique))
    revised, dt3, in3, out3 = run_ar(tok, model, rprompt, **revise_cfg)
    revised = ensure_end_line(revised)

    meta = {
        "draft": {"latency_s": dt1, "input_tokens": in1, "output_tokens": out1},
        "critique": {"latency_s": dt2, "input_tokens": in2, "output_tokens": out2},
        "revise": {"latency_s": dt3, "input_tokens": in3, "output_tokens": out3},
        "total_latency_s": (dt1 or 0) + (dt2 or 0) + (dt3 or 0),
        "total_output_tokens": (out1 or 0) + (out2 or 0) + (out3 or 0),
    }
    return revised, meta, draft, critique


# Convenience wrappers for run_agent_loop
def run_mdlm_runner(tok, model, prompt, **cfg):
    return run_mdlm_quickstart(tok, model, prompt, **cfg)


def run_bd3lm_runner(tok, model, prompt, **cfg):
    return run_bd3lm_quickstart(tok, model, prompt, **cfg)


def run_wedlm_runner(tok, llm, prompt, **cfg):
    return run_wedlm(tok, llm, prompt, **cfg)


# =========================
# 9) In-env DAG (AR <-> BD3LM)
# =========================

def run_ar_then_bd3lm(
    ar_tok, ar_model,
    bd_tok, bd_model,
    prompt: str,
    plan_cfg=None,
    write_cfg=None,
):
    plan_cfg = plan_cfg or {"max_new_tokens": 192, "temperature": 0.7, "top_p": 0.95}
    write_cfg = write_cfg or {
        "max_new_tokens": 256,
        "steps": 256,
        "block_size": 32,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
    }

    plan_prompt = (
        f"{prompt}\n\n"
        "[TASK]\nReturn ONLY valid JSON with keys:\n"
        "goal (string <= 18 words),\n"
        "controls (array of exactly 3 bullets),\n"
        "readout (array of exactly 3 bullets),\n"
        "analysis (array of exactly 3 bullets),\n"
        "pitfalls (array of exactly 3 bullets).\n"
        "Constraints:\n"
        "- Each bullet <= 12 words.\n"
        "- No proper nouns or exotic systems unless required.\n"
        "- No repeated phrases.\n"
    )

    plan_text, dt1, in1, out1 = run_ar(ar_tok, ar_model, plan_prompt, **plan_cfg)

    write_prompt = (
        f"{prompt}\n\n"
        "[PLAN_JSON]\n{plan_text}\n\n"
        "[TASK]\nWrite the final answer with the exact headings:\n"
        "Goal, Controls, Readout, Analysis, Pitfalls.\n"
        "Use ONLY the bullets from PLAN_JSON.\n"
        "You may add at most 1 short sentence under Goal and at most 6 extra words per bullet.\n"
        "Do not introduce new technologies not in PLAN_JSON.\n"
        f"End with {END_MARK}."
    )

    final_text, dt2, in2, out2 = run_bd3lm_quickstart(bd_tok, bd_model, write_prompt, **write_cfg)

    meta = {
        "plan_text": plan_text,
        "plan_cfg": plan_cfg,
        "write_cfg": write_cfg,
        "stage_latencies_s": {"ar_plan": dt1, "bd3lm_write": dt2},
        "stage_output_tokens": {"ar_plan": out1, "bd3lm_write": out2},
    }
    return final_text, (dt1 + dt2), (in1 + in2), (out1 + out2), meta


def run_bd3lm_then_ar(
    bd_tok, bd_model,
    ar_tok, ar_model,
    prompt: str,
    draft_cfg=None,
    refine_cfg=None,
):
    draft_cfg = draft_cfg or {
        "max_new_tokens": 256,
        "steps": 256,
        "block_size": 32,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
    }
    refine_cfg = refine_cfg or {"max_new_tokens": 256, "temperature": 0.6, "top_p": 0.95}

    draft_prompt = f"{prompt}\n\n[TASK]\nWrite a complete first draft answer."
    draft_text, dt1, in1, out1 = run_bd3lm_quickstart(bd_tok, bd_model, draft_prompt, **draft_cfg)

    refine_prompt = (
        f"{prompt}\n\n"
        "[DRAFT]\n{draft_text}\n\n"
        "[TASK]\nRewrite the draft for clarity, completeness, and good structure. "
        "Fix any unfinished sentences, contradictions, or missing pieces. "
        "Return only the final answer."
    )
    final_text, dt2, in2, out2 = run_ar(ar_tok, ar_model, refine_prompt, **refine_cfg)

    meta = {
        "draft_text": draft_text,
        "draft_cfg": draft_cfg,
        "refine_cfg": refine_cfg,
        "stage_latencies_s": {"bd3lm_draft": dt1, "ar_refine": dt2},
        "stage_output_tokens": {"bd3lm_draft": out1, "ar_refine": out2},
    }
    return final_text, (dt1 + dt2), (in1 + in2), (out1 + out2), meta


# =========================
# 10) Cross-env DAG buffer tooling (AR8B <-> WeDLM8B)
# =========================

def _dag_refine_prompt(base_prompt: str, draft_text: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "[DRAFT]\n"
        f"{draft_text}\n\n"
        "[TASK]\n"
        "Rewrite the draft to satisfy the task prompt constraints EXACTLY.\n"
        "- Preserve the same headings and bullet requirements.\n"
        "- Fix missing headings, wrong bullet counts, repetition, and missing <END>.\n"
        "- Keep content as close as possible; only change what is needed for compliance.\n"
        f"- End with {END_MARK} on its own line.\n"
        "Return ONLY the final corrected answer."
    )


def _dag_plan_json_prompt(base_prompt: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "[TASK]\nReturn ONLY valid JSON with keys:\n"
        "goal (string <= 18 words),\n"
        "controls (array of exactly 3 bullets),\n"
        "readout (array of exactly 3 bullets),\n"
        "analysis (array of exactly 3 bullets),\n"
        "pitfalls (array of exactly 3 bullets).\n"
        "Constraints:\n"
        "- Each bullet <= 12 words.\n"
        "- No repeated phrases.\n"
        "- No extra keys.\n"
    )


def _dag_write_from_plan_prompt(base_prompt: str, plan_json: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "[PLAN_JSON]\n{plan_json}\n\n"
        "[TASK]\nWrite the final answer with the exact headings:\n"
        "Goal, Controls, Readout, Analysis, Pitfalls.\n"
        "Use ONLY the bullets from PLAN_JSON.\n"
        "You may add at most 1 short sentence under Goal and at most 6 extra words per bullet.\n"
        "Do not introduce new technologies not in PLAN_JSON.\n"
        f"End with {END_MARK} on its own line."
    )


def dag_ar8b_to_wedlm8b_make_buffer(
    prompts: List[dict],
    style: str,
    ar_tok, ar_model,
    out_buf_path: str,
    plan_cfg=None,
    use_plan: bool = True,
):
    plan_cfg = plan_cfg or {"max_new_tokens": 192, "temperature": 0.7, "top_p": 0.95}
    draft_cfg = {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.95}

    rows = []
    for item in prompts:
        pid = item.get("id", "")
        base_prompt = item["prompt"]
        prompt = apply_prompt_style(base_prompt, style)

        if use_plan:
            plan_prompt = _dag_plan_json_prompt(prompt)
            plan_text, dt1, in1, out1 = run_ar(ar_tok, ar_model, plan_prompt, **plan_cfg)
            stage2 = {
                "mode": "plan_write",
                "plan_prompt": plan_prompt,
                "plan_text": plan_text,
                "write_prompt": _dag_write_from_plan_prompt(prompt, plan_text),
            }
            stage1_resp = plan_text
        else:
            draft_prompt = f"{prompt}\n\n[TASK]\nWrite a complete first draft answer. End with {END_MARK}."
            stage1_resp, dt1, in1, out1 = run_ar(ar_tok, ar_model, draft_prompt, **draft_cfg)
            stage2 = {"mode": "refine", "refine_prompt": _dag_refine_prompt(prompt, stage1_resp)}

        rows.append(
            {
                "dag_type": "ar8b_to_wedlm8b",
                "prompt_id": pid,
                "prompt_style": style,
                "prompt": prompt,
                "stage1": {
                    "model_tag": "ar_8b",
                    "model_id": MODELS["ar_8b"],
                    "response": stage1_resp,
                    "latency_s": dt1,
                    "input_tokens": in1,
                    "output_tokens": out1,
                    "meta": {"use_plan": use_plan},
                },
                "stage2_request": stage2,
            }
        )

    write_jsonl(out_buf_path, rows)
    print(f"[DAG] Wrote buffer: {out_buf_path} ({len(rows)} rows)")


def dag_wedlm8b_to_ar8b_make_buffer(
    prompts: List[dict],
    style: str,
    wedlm_tok, wedlm_llm,
    out_buf_path: str,
    draft_cfg=None,
):
    draft_cfg = draft_cfg or {"max_new_tokens": 256, "temperature": 0.0}

    rows = []
    for item in prompts:
        pid = item.get("id", "")
        base_prompt = item["prompt"]
        prompt = apply_prompt_style(base_prompt, style)

        draft_prompt = f"{prompt}\n\n[TASK]\nWrite a complete first draft answer. End with {END_MARK}."
        draft_text, dt1, in1, out1 = run_wedlm(wedlm_tok, wedlm_llm, draft_prompt, **draft_cfg)

        rows.append(
            {
                "dag_type": "wedlm8b_to_ar8b",
                "prompt_id": pid,
                "prompt_style": style,
                "prompt": prompt,
                "stage1": {
                    "model_tag": "wedlm_8b",
                    "model_id": MODELS["diff8"],
                    "response": draft_text,
                    "latency_s": dt1,
                    "input_tokens": in1,
                    "output_tokens": out1,
                    "meta": {"draft_cfg": draft_cfg},
                },
                "stage2_request": {"mode": "refine", "refine_prompt": _dag_refine_prompt(prompt, draft_text)},
            }
        )

    write_jsonl(out_buf_path, rows)
    print(f"[DAG] Wrote buffer: {out_buf_path} ({len(rows)} rows)")


def dag_ar8b_consume_buffer_and_write_results(
    buf_path: str,
    out_path: str,
    ar_tok, ar_model,
    append: bool = False,
    refine_cfg=None,
    dag_tag: Optional[str] = None,
):
    refine_cfg = refine_cfg or {"max_new_tokens": 256, "temperature": 0.2, "top_p": 1.0}
    mode = "a" if append else "w"

    with open(out_path, mode) as fout:
        for row in iter_jsonl(buf_path):
            assert row["dag_type"] == "wedlm8b_to_ar8b", f"Expected wedlm8b_to_ar8b, got {row.get('dag_type')}"
            pid = row["prompt_id"]
            style = row["prompt_style"]
            prompt = row["prompt"]

            refine_prompt = row["stage2_request"]["refine_prompt"]
            final_text, dt2, in2, out2 = run_ar(ar_tok, ar_model, refine_prompt, **refine_cfg)

            total_dt = (row["stage1"].get("latency_s") or 0) + (dt2 or 0)
            total_out = (row["stage1"].get("output_tokens") or 0) + (out2 or 0)
            total_in = (row["stage1"].get("input_tokens") or 0) + (in2 or 0)

            fout.write(
                json.dumps(
                    {
                        "model_tag": dag_tag or "wedlm8b_ar8b_refine",
                        "model_id": {"stage1": row["stage1"]["model_id"], "stage2": MODELS["ar_8b"]},
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": final_text,
                        "latency_s": total_dt,
                        "input_tokens": total_in,
                        "output_tokens": total_out,
                        "tokens_per_sec": (total_out / total_dt) if total_dt and total_dt > 0 else None,
                        "meta": {"dag_type": row["dag_type"], "stage1": row["stage1"], "stage2_cfg": refine_cfg},
                    }
                )
                + "\n"
            )
            fout.flush()

    print(f"[DAG] Wrote results: {out_path} (from {buf_path})")


def dag_wedlm8b_consume_buffer_and_write_results(
    buf_path: str,
    out_path: str,
    wedlm_tok, wedlm_llm,
    append: bool = False,
    write_cfg=None,
    refine_cfg=None,
    dag_tag: Optional[str] = None,
):
    write_cfg = write_cfg or {"max_new_tokens": 256, "temperature": 0.0}
    refine_cfg = refine_cfg or {"max_new_tokens": 256, "temperature": 0.0}
    mode = "a" if append else "w"

    with open(out_path, mode) as fout:
        for row in iter_jsonl(buf_path):
            assert row["dag_type"] == "ar8b_to_wedlm8b", f"Expected ar8b_to_wedlm8b, got {row.get('dag_type')}"
            pid = row["prompt_id"]
            style = row["prompt_style"]
            prompt = row["prompt"]
            req = row["stage2_request"]

            if req["mode"] == "plan_write":
                stage2_prompt = req["write_prompt"]
                final_text, dt2, in2, out2 = run_wedlm(wedlm_tok, wedlm_llm, stage2_prompt, **write_cfg)
                stage2_cfg = write_cfg
            elif req["mode"] == "refine":
                stage2_prompt = req["refine_prompt"]
                final_text, dt2, in2, out2 = run_wedlm(wedlm_tok, wedlm_llm, stage2_prompt, **refine_cfg)
                stage2_cfg = refine_cfg
            else:
                raise ValueError(f"Unknown stage2_request mode: {req.get('mode')}")

            total_dt = (row["stage1"].get("latency_s") or 0) + (dt2 or 0)
            total_out = (row["stage1"].get("output_tokens") or 0) + (out2 or 0)
            total_in = (row["stage1"].get("input_tokens") or 0) + (in2 or 0)

            fout.write(
                json.dumps(
                    {
                        "model_tag": dag_tag or "ar8b_wedlm8b_refine",
                        "model_id": {"stage1": row["stage1"]["model_id"], "stage2": MODELS["diff8"]},
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": final_text,
                        "latency_s": total_dt,
                        "input_tokens": total_in,
                        "output_tokens": total_out,
                        "tokens_per_sec": (total_out / total_dt) if total_dt and total_dt > 0 else None,
                        "meta": {
                            "dag_type": row["dag_type"],
                            "stage1": row["stage1"],
                            "stage2_cfg": stage2_cfg,
                            "stage2_mode": req["mode"],
                        },
                    }
                )
                + "\n"
            )
            fout.flush()

    print(f"[DAG] Wrote results: {out_path} (from {buf_path})")


# =========================
# 11) Default configs
# =========================

WEDLM_AGENT = {
    "gen": {"max_new_tokens": 256, "temperature": 0.0},
    "critique": {"max_new_tokens": 128, "temperature": 0.0},
    "revise": {"max_new_tokens": 256, "temperature": 0.0},
}

MDLM_AGENT = {
    "gen": {"max_new_tokens": 256, "steps": 256, "block_size": 8, "temperature": 0.1, "cfg_scale": 0.0, "remasking": "low_confidence"},
    "critique": {"max_new_tokens": 128, "steps": 64, "block_size": 8, "temperature": 0.0, "cfg_scale": 0.0, "remasking": "low_confidence"},
    "revise": {"max_new_tokens": 256, "steps": 128, "block_size": 8, "temperature": 0.1, "cfg_scale": 0.0, "remasking": "low_confidence"},
}

BD3LM_AGENT = {
    "gen": {"max_new_tokens": 256, "steps": 256, "block_size": 32, "temperature": 0.0, "cfg_scale": 0.0, "remasking": "low_confidence"},
    "critique": {"max_new_tokens": 128, "steps": 64, "block_size": 32, "temperature": 0.0, "cfg_scale": 0.0, "remasking": "low_confidence"},
    "revise": {"max_new_tokens": 256, "steps": 128, "block_size": 32, "temperature": 0.0, "cfg_scale": 0.0, "remasking": "low_confidence"},
}


# =========================
# 12) Bench entrypoint
# =========================

def bench(args) -> None:
    prompts = load_prompts(args.prompts, limit=args.limit)
    out_path = args.out
    mode = "a" if args.append else "w"
    style = "s1"

    if args.dag_dir:
        ensure_dir(args.dag_dir)

    # -------------------------
    # Target: WeDLM env
    # -------------------------
    if args.target == "wedlm":
        wedlm_tok, wedlm_llm = load_wedlm_engine(MODELS["diff8"])

        if args.dag_mode == "make_buffer":
            buf_path = args.buf or os.path.join(args.dag_dir, "wedlm8b_to_ar8b_buf.jsonl")
            dag_wedlm8b_to_ar8b_make_buffer(
                prompts=prompts,
                style=style,
                wedlm_tok=wedlm_tok,
                wedlm_llm=wedlm_llm,
                out_buf_path=buf_path,
                draft_cfg={"max_new_tokens": 256, "temperature": 0.0},
            )
            return

        if args.dag_mode == "consume_buffer":
            if not args.buf:
                raise ValueError("--buf is required for consume_buffer")
            dag_wedlm8b_consume_buffer_and_write_results(
                buf_path=args.buf,
                out_path=out_path,
                wedlm_tok=wedlm_tok,
                wedlm_llm=wedlm_llm,
                append=args.append,
                write_cfg={"max_new_tokens": 256, "temperature": 0.0},
                refine_cfg={"max_new_tokens": 256, "temperature": 0.0},
                dag_tag=args.dag_tag,
            )
            return

        with open(out_path, mode) as fout:
            for item in prompts:
                pid = item.get("id", "")
                base_prompt = item["prompt"]
                prompt = apply_prompt_style(base_prompt, style)

                # ---- WeDLM base
                text, dt, in_toks, out_toks = run_wedlm(
                    wedlm_tok, wedlm_llm, prompt, max_new_tokens=256, temperature=0.0
                )
                fout.write(
                    json.dumps(
                        {
                            "model_tag": "wedlm_8b",
                            "model_id": MODELS["diff8"],
                            "prompt_id": pid,
                            "prompt_style": style,
                            "prompt": prompt,
                            "response": text,
                            "latency_s": dt,
                            "input_tokens": in_toks,
                            "output_tokens": out_toks,
                            "tokens_per_sec": (out_toks / dt) if dt and dt > 0 else None,
                        }
                    )
                    + "\n"
                )
                fout.flush()

                # ---- WeDLM + agent
                agent_text, agent_meta, draft, critique = run_agent_loop(
                    run_wedlm_runner,
                    wedlm_tok,
                    wedlm_llm,
                    prompt,
                    gen_cfg=WEDLM_AGENT["gen"],
                    critique_cfg=WEDLM_AGENT["critique"],
                    revise_cfg=WEDLM_AGENT["revise"],
                )
                fout.write(
                    json.dumps(
                        {
                            "model_tag": "wedlm_8b_agent",
                            "model_id": MODELS["diff8"],
                            "prompt_id": pid,
                            "prompt_style": style,
                            "prompt": prompt,
                            "response": agent_text,
                            "agent": {
                                "type": "draft_critique_revise",
                                "draft": strip_after_end(draft),
                                "critique": strip_after_end(critique),
                                "meta": agent_meta,
                            },
                            "latency_s": agent_meta["total_latency_s"],
                            "output_tokens": agent_meta["total_output_tokens"],
                            "tokens_per_sec": (
                                agent_meta["total_output_tokens"] / agent_meta["total_latency_s"]
                                if agent_meta["total_latency_s"] > 0
                                else None
                            ),
                        }
                    )
                    + "\n"
                )
                fout.flush()

        print(f"Wrote {out_path}")
        return

    # -------------------------
    # Target: NOFLASH env
    # -------------------------
    ar8b_tok, ar8b_model = load_ar(MODELS["ar_8b"])

    ar_tok, ar_model = load_ar(MODELS["ar"])
    mdlm_tok, mdlm_model = load_mlm(MODELS["mdlm"])
    bd3lm_tok, bd3lm_model = load_mlm(MODELS["bd3lm"])

    # Some diffusion checkpoints require resizing embeddings to tokenizer size
    mdlm_model.resize_token_embeddings(len(mdlm_tok))
    bd3lm_model.resize_token_embeddings(len(bd3lm_tok))

    if args.dag_mode == "make_buffer":
        buf_path = args.buf or os.path.join(args.dag_dir, "ar8b_to_wedlm8b_buf.jsonl")
        dag_ar8b_to_wedlm8b_make_buffer(
            prompts=prompts,
            style=style,
            ar_tok=ar8b_tok,
            ar_model=ar8b_model,
            out_buf_path=buf_path,
            plan_cfg={"max_new_tokens": 192, "temperature": 0.7, "top_p": 0.95},
            use_plan=True,
        )
        return

    if args.dag_mode == "consume_buffer":
        if not args.buf:
            raise ValueError("--buf is required for consume_buffer")
        dag_ar8b_consume_buffer_and_write_results(
            buf_path=args.buf,
            out_path=out_path,
            ar_tok=ar8b_tok,
            ar_model=ar8b_model,
            append=args.append,
            refine_cfg={"max_new_tokens": 256, "temperature": 0.2, "top_p": 1.0},
            dag_tag=args.dag_tag,
        )
        return

    with open(out_path, mode) as fout:
        for item in prompts:
            pid = item.get("id", "")
            base_prompt = item["prompt"]
            prompt = apply_prompt_style(base_prompt, style)

            # ---- AR 8B agent
            agent_text, agent_meta, draft, critique = run_ar_agent_loop(
                ar8b_tok, ar8b_model, prompt, gen_cfg={"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.95}
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "ar_8b_agent",
                        "model_id": MODELS["ar_8b"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": agent_text,
                        "agent": {
                            "type": "draft_critique_revise",
                            "draft": strip_after_end(draft),
                            "critique": strip_after_end(critique),
                            "meta": agent_meta,
                        },
                        "latency_s": agent_meta["total_latency_s"],
                        "output_tokens": agent_meta["total_output_tokens"],
                        "tokens_per_sec": (
                            agent_meta["total_output_tokens"] / agent_meta["total_latency_s"]
                            if agent_meta["total_latency_s"] > 0
                            else None
                        ),
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- AR 8B base
            text, dt, in_toks, out_toks = run_ar(
                ar8b_tok, ar8b_model, prompt, max_new_tokens=256, temperature=0.7, top_p=0.95
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "ar_8b",
                        "model_id": MODELS["ar_8b"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": text,
                        "latency_s": dt,
                        "input_tokens": in_toks,
                        "output_tokens": out_toks,
                        "tokens_per_sec": (out_toks / dt) if dt and dt > 0 else None,
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- AR base
            text, dt, in_toks, out_toks = run_ar(
                ar_tok, ar_model, prompt, max_new_tokens=256, temperature=0.7, top_p=0.95
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "ar",
                        "model_id": MODELS["ar"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": text,
                        "latency_s": dt,
                        "input_tokens": in_toks,
                        "output_tokens": out_toks,
                        "tokens_per_sec": (out_toks / dt) if dt and dt > 0 else None,
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- AR agent
            agent_text, agent_meta, draft, critique = run_ar_agent_loop(
                ar_tok, ar_model, prompt, gen_cfg={"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.95}
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "ar_agent",
                        "model_id": MODELS["ar"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": agent_text,
                        "agent": {
                            "type": "draft_critique_revise",
                            "draft": strip_after_end(draft),
                            "critique": strip_after_end(critique),
                            "meta": agent_meta,
                        },
                        "latency_s": agent_meta["total_latency_s"],
                        "input_tokens": None,
                        "output_tokens": agent_meta["total_output_tokens"],
                        "tokens_per_sec": (
                            agent_meta["total_output_tokens"] / agent_meta["total_latency_s"]
                            if agent_meta["total_latency_s"] > 0
                            else None
                        ),
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- MDLM base
            text, dt, in_toks, out_toks = run_mdlm_quickstart(
                mdlm_tok,
                mdlm_model,
                prompt,
                max_new_tokens=256,
                steps=256,
                block_size=16,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "mdlm",
                        "model_id": MODELS["mdlm"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": text,
                        "latency_s": dt,
                        "input_tokens": in_toks,
                        "output_tokens": out_toks,
                        "tokens_per_sec": (out_toks / dt) if dt and dt > 0 else None,
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- MDLM agent
            agent_text, agent_meta, draft, critique = run_agent_loop(
                run_mdlm_runner,
                mdlm_tok,
                mdlm_model,
                prompt,
                gen_cfg=MDLM_AGENT["gen"],
                critique_cfg=MDLM_AGENT["critique"],
                revise_cfg=MDLM_AGENT["revise"],
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "mdlm_agent",
                        "model_id": MODELS["mdlm"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": agent_text,
                        "agent": {
                            "type": "draft_critique_revise",
                            "draft": strip_after_end(draft),
                            "critique": strip_after_end(critique),
                            "meta": agent_meta,
                        },
                        "latency_s": agent_meta["total_latency_s"],
                        "output_tokens": agent_meta["total_output_tokens"],
                        "tokens_per_sec": (
                            agent_meta["total_output_tokens"] / agent_meta["total_latency_s"]
                            if agent_meta["total_latency_s"] > 0
                            else None
                        ),
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- BD3LM base (no finish/continue hacks)
            text, dt, in_toks, out_toks = run_bd3lm_quickstart(
                bd3lm_tok,
                bd3lm_model,
                prompt,
                max_new_tokens=256,
                steps=256,
                block_size=32,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "bd3lm",
                        "model_id": MODELS["bd3lm"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": text,
                        "latency_s": dt,
                        "input_tokens": in_toks,
                        "output_tokens": out_toks,
                        "tokens_per_sec": (out_toks / dt) if dt and dt > 0 else None,
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- BD3LM agent (no finish/continue hacks)
            agent_text, agent_meta, draft, critique = run_agent_loop(
                run_bd3lm_runner,
                bd3lm_tok,
                bd3lm_model,
                prompt,
                gen_cfg=BD3LM_AGENT["gen"],
                critique_cfg=BD3LM_AGENT["critique"],
                revise_cfg=BD3LM_AGENT["revise"],
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "bd3lm_agent",
                        "model_id": MODELS["bd3lm"],
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": agent_text,
                        "agent": {
                            "type": "draft_critique_revise",
                            "draft": strip_after_end(draft),
                            "critique": strip_after_end(critique),
                            "meta": agent_meta,
                        },
                        "latency_s": agent_meta["total_latency_s"],
                        "output_tokens": agent_meta["total_output_tokens"],
                        "tokens_per_sec": (
                            agent_meta["total_output_tokens"] / agent_meta["total_latency_s"]
                            if agent_meta["total_latency_s"] > 0
                            else None
                        ),
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- In-env DAG: AR -> BD3LM
            dag_text, dag_dt, dag_in, dag_out, dag_meta = run_ar_then_bd3lm(
                ar_tok,
                ar_model,
                bd3lm_tok,
                bd3lm_model,
                prompt,
                plan_cfg={"max_new_tokens": 192, "temperature": 0.7, "top_p": 0.95},
                write_cfg={
                    "max_new_tokens": 256,
                    "steps": 256,
                    "block_size": 32,
                    "temperature": 0.0,
                    "cfg_scale": 0.0,
                    "remasking": "low_confidence",
                },
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "ar_bd3lm",
                        "model_id": {"ar": MODELS["ar"], "bd3lm": MODELS["bd3lm"]},
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": dag_text,
                        "latency_s": dag_dt,
                        "input_tokens": dag_in,
                        "output_tokens": dag_out,
                        "tokens_per_sec": (dag_out / dag_dt) if dag_dt and dag_dt > 0 else None,
                        "meta": dag_meta,
                    }
                )
                + "\n"
            )
            fout.flush()

            # ---- In-env DAG: BD3LM -> AR
            dag_text, dag_dt, dag_in, dag_out, dag_meta = run_bd3lm_then_ar(
                bd3lm_tok,
                bd3lm_model,
                ar_tok,
                ar_model,
                prompt,
                draft_cfg={
                    "max_new_tokens": 256,
                    "steps": 256,
                    "block_size": 32,
                    "temperature": 0.0,
                    "cfg_scale": 0.0,
                    "remasking": "low_confidence",
                },
                refine_cfg={"max_new_tokens": 256, "temperature": 0.6, "top_p": 0.95},
            )
            fout.write(
                json.dumps(
                    {
                        "model_tag": "bd3lm_ar",
                        "model_id": {"bd3lm": MODELS["bd3lm"], "ar": MODELS["ar"]},
                        "prompt_id": pid,
                        "prompt_style": style,
                        "prompt": prompt,
                        "response": dag_text,
                        "latency_s": dag_dt,
                        "input_tokens": dag_in,
                        "output_tokens": dag_out,
                        "tokens_per_sec": (dag_out / dag_dt) if dag_dt and dag_dt > 0 else None,
                        "meta": dag_meta,
                    }
                )
                + "\n"
            )
            fout.flush()

    print(f"Wrote {out_path}")


# =========================
# 13) Main
# =========================

if __name__ == "__main__":
    args = parse_args()
    print(f"DEVICE={DEVICE} DTYPE={DTYPE}")
    bench(args)
