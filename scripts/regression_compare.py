#!/usr/bin/env python3
"""Run small generation probes against llmtoy and optional reference engines.

This is intentionally light infrastructure, not a full benchmark harness. It
captures commands, stdout, stderr, exit codes, and elapsed time as JSON so we can
start building repeatable correctness checks before doing more optimization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_command(name: str, cmd: list[str], timeout_s: int) -> dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "command": cmd,
            "exit_code": proc.returncode,
            "elapsed_s": round(elapsed, 3),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "command": cmd,
            "exit_code": None,
            "elapsed_s": round(elapsed, 3),
            "timeout": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def find_llama_cli(explicit: str | None) -> list[str] | None:
    if explicit:
        return [explicit]
    for candidate in (
        "llama-cli",
        "/opt/ai-lab/llama.cpp/build/bin/llama-cli",
        "/opt/ai-lab/llama.cpp/build/bin/main",
    ):
        found = shutil.which(candidate) if "/" not in candidate else candidate
        if found and Path(found).exists():
            return [found]
    if shutil.which("nix"):
        return ["nix", "shell", "nixpkgs#llama-cpp", "-c", "llama-cli"]
    return None


def run_transformers(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional deps
        return {
            "name": "transformers",
            "skipped": True,
            "reason": f"optional dependency unavailable: {exc}",
        }

    try:  # pragma: no cover - depends on optional deps/model
        tokenizer = AutoTokenizer.from_pretrained(args.transformers_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.transformers_model,
            torch_dtype="auto",
            device_map="auto",
        )
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-6),
        )
        text = tokenizer.decode(output[0], skip_special_tokens=False)
        return {
            "name": "transformers",
            "exit_code": 0,
            "elapsed_s": round(time.monotonic() - start, 3),
            "stdout": text,
            "stderr": "",
        }
    except Exception as exc:
        return {
            "name": "transformers",
            "exit_code": 1,
            "elapsed_s": round(time.monotonic() - start, 3),
            "stdout": "",
            "stderr": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GGUF model path for llmtoy/llama.cpp")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--llmtoy", default="./zig-out/bin/llmtoy")
    parser.add_argument("--llama-cli", default=None, help="Path to llama.cpp llama-cli")
    parser.add_argument("--transformers-model", default=None, help="HF model id/path")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat", action="store_true", help="Pass --chat to llmtoy")
    parser.add_argument("--expect-substring", default=None)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []

    llmtoy_cmd = [
        args.llmtoy,
        "generate",
        args.model,
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--threads",
        str(args.threads),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]
    if args.chat:
        llmtoy_cmd.insert(4, "--chat")
    results.append(run_command("llmtoy", llmtoy_cmd, args.timeout_s))

    llama_cmd_base = find_llama_cli(args.llama_cli)
    if llama_cmd_base:
        llama_cmd = [
            *llama_cmd_base,
            "-m",
            args.model,
            "-p",
            args.prompt,
            "-n",
            str(args.max_tokens),
            "-t",
            str(args.threads),
            "--temp",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--seed",
            str(args.seed),
            "--no-display-prompt",
            "--no-warmup",
        ]
        if args.chat:
            llama_cmd.extend(["--conversation", "--single-turn", "--reasoning", "off"])
        results.append(run_command("llama.cpp", llama_cmd, args.timeout_s))
    else:
        results.append({"name": "llama.cpp", "skipped": True, "reason": "llama-cli not found"})

    if args.transformers_model:
        results.append(run_transformers(args))

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "threads": args.threads,
        "results": results,
    }

    ok = True
    for result in results:
        if result.get("skipped"):
            continue
        if result.get("exit_code") != 0:
            ok = False
        if args.expect_substring and args.expect_substring not in result.get("stdout", ""):
            result["expectation_failed"] = f"missing substring: {args.expect_substring}"
            ok = False

    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
