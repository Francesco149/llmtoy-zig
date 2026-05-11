#!/usr/bin/env python3
"""Run small generation probes against llmtoy and optional reference engines.

This is intentionally light infrastructure, not a full benchmark harness. It
captures commands, stdout, stderr, exit codes, and elapsed time as JSON so we can
start building repeatable correctness checks before doing more optimization.
"""

from __future__ import annotations

import argparse
import json
import re
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


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LLMTOY_SETUP_RE = re.compile(r"setup:\s+(\d+)\s+ms")
LLMTOY_TOKEN_TIMING_RE = re.compile(
    r"(prefill|generation):\s+(\d+)\s+tokens\s+in\s+(\d+)\s+ms\s+\(([\d.]+)\s+tok/s\)"
)
LLAMA_TPS_RE = re.compile(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]")


def clean_text(text: str) -> str:
    text = text.replace("\b", "")
    text = ANSI_RE.sub("", text)
    # llama.cpp progress indicators can rewrite a terminal line with \r. Keep
    # only the final visible segment for each such line.
    lines = []
    for line in text.splitlines():
        lines.append(line.split("\r")[-1])
    return "\n".join(lines).strip()


def generation_text(result: dict[str, Any]) -> str:
    text = clean_text(result.get("stdout", ""))
    if result.get("name") == "llmtoy":
        # In chat mode llmtoy echoes the rendered prompt to stdout before tokens.
        # The answer begins after Gemma's empty thought-channel close marker.
        if "<channel|>" in text:
            return text.split("<channel|>", 1)[1].strip()
        if "<|turn>model" in text:
            return text.split("<|turn>model", 1)[1].strip()
    if result.get("name") == "llama.cpp":
        # llama-cli conversation mode can print an interactive banner and prompt
        # even with logging disabled. The assistant text follows the last user
        # prompt line that starts with "> ".
        prompt_markers = [i for i, line in enumerate(text.splitlines()) if line.startswith("> ")]
        lines = text.splitlines()
        if prompt_markers:
            text = "\n".join(lines[prompt_markers[-1] + 1 :])
        text = re.sub(r"^[\s|/\\-]+", "", text)
        text = re.sub(r"\n\[ Prompt:.*", "", text, flags=re.S)
        text = text.replace("Exiting...", "")
    return text


def extract_timings(result: dict[str, Any]) -> dict[str, Any]:
    text = clean_text(result.get("stderr", "") + "\n" + result.get("stdout", ""))
    timings: dict[str, Any] = {}

    if result.get("name") == "llmtoy":
        if match := LLMTOY_SETUP_RE.search(text):
            timings["setup_ms"] = int(match.group(1))
        for match in LLMTOY_TOKEN_TIMING_RE.finditer(text):
            label = match.group(1)
            timings[label] = {
                "tokens": int(match.group(2)),
                "ms": int(match.group(3)),
                "tok_s": float(match.group(4)),
            }

    if result.get("name") == "llama.cpp":
        if match := LLAMA_TPS_RE.search(text):
            timings["prefill_tok_s"] = float(match.group(1))
            timings["generation_tok_s"] = float(match.group(2))

    return timings


def print_human_report(report: dict[str, Any]) -> None:
    print("Regression comparison")
    print(f"model: {report['model']}")
    print(f"prompt: {report['prompt']}")
    print(
        "settings: "
        f"tokens={report['max_tokens']} threads={report['threads']} "
        f"temp={report['temperature']} top_k={report['top_k']} top_p={report['top_p']}"
    )
    print()

    for result in report["results"]:
        name = result["name"]
        if result.get("skipped"):
            print(f"== {name}: skipped ==")
            print(result.get("reason", ""))
            print()
            continue

        status = result.get("exit_code")
        elapsed = result.get("elapsed_s")
        print(f"== {name}: exit={status} elapsed={elapsed}s ==")
        if result.get("expectation_failed"):
            print(result["expectation_failed"])

        timings = result.get("timings") or {}
        if timings:
            print("-- timings --")
            if "setup_ms" in timings:
                print(f"setup: {timings['setup_ms']} ms")
            if "prefill" in timings:
                prefill = timings["prefill"]
                print(f"prefill: {prefill['tokens']} tokens in {prefill['ms']} ms ({prefill['tok_s']:.2f} tok/s)")
            if "generation" in timings:
                gen = timings["generation"]
                print(f"generation: {gen['tokens']} tokens in {gen['ms']} ms ({gen['tok_s']:.2f} tok/s)")
            if "prefill_tok_s" in timings:
                print(f"prefill: {timings['prefill_tok_s']:.2f} tok/s")
            if "generation_tok_s" in timings:
                print(f"generation: {timings['generation_tok_s']:.2f} tok/s")

        gen = generation_text(result)
        print("-- generation --")
        print(gen if gen else "(no stdout)")

        stderr = clean_text(result.get("stderr", ""))
        if stderr and status != 0:
            print("-- stderr --")
            print(stderr[-2000:])
        print()


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
    parser.add_argument("--json-only", action="store_true")
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
            "--log-disable",
            "--simple-io",
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
        result["timings"] = extract_timings(result)
        if result.get("exit_code") != 0:
            ok = False
        if args.expect_substring and args.expect_substring not in result.get("stdout", ""):
            result["expectation_failed"] = f"missing substring: {args.expect_substring}"
            ok = False

    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")
    if args.json_only:
        print(encoded)
    else:
        print_human_report(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
