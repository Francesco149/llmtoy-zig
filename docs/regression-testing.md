# Regression Testing Against Reference Engines

The goal is to compare `llmtoy` against known-good engines before doing more
optimization. The first harness is intentionally small: it prints a compact
human-readable comparison by default and can also emit JSON for fixtures.

Run one model/engine comparison at a time. These CPU runs are slow and memory
hungry, and parallel reference runs can make failures look like correctness
problems when they are really resource contention.

When using this as a benchmark, check that the host is quiet before every run:

```sh
scripts/check_benchmark_noise.sh
```

```sh
nix develop --command python3 scripts/regression_compare.py \
  --model /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  --prompt "Briefly explain the full forward pass of a MoE model" \
  --chat \
  --max-tokens 32 \
  --threads 12 \
  --temperature 0.1 \
  --expect-substring MoE
```

The script always runs `./zig-out/bin/llmtoy`. If `llama-cli` is available in
`PATH`, or at `/opt/ai-lab/llama.cpp/build/bin/llama-cli`, it also runs
llama.cpp. Otherwise it falls back to:

```sh
nix shell nixpkgs#llama-cpp -c llama-cli ...
```

A custom path can be passed with `--llama-cli`.

By default the output is designed for visual inspection:

```text
== llmtoy: exit=0 elapsed=...s ==
-- generation --
The forward pass of a Mixture-of-Experts ...

== llama.cpp: exit=0 elapsed=...s ==
-- generation --
The forward pass of a Mixture of Experts ...
```

Use `--json-only` to print the machine-readable JSON record:

```json
{
  "model": "...",
  "prompt": "...",
  "results": [
    {
      "name": "llmtoy",
      "command": ["./zig-out/bin/llmtoy", "..."],
      "exit_code": 0,
      "elapsed_s": 12.3,
      "stdout": "...",
      "stderr": "..."
    }
  ]
}
```

Use `--json-out path` to save the JSON report while still printing the
human-readable comparison.

## llama.cpp Notes

The script assumes modern `llama-cli` flags:

```text
-m <model> -p <prompt> -n <tokens> -t <threads> --temp <temperature> --seed <seed> --no-display-prompt --log-disable
```

If the local llama.cpp clone changes CLI flags, update
`scripts/regression_compare.py` rather than encoding those details in test docs.

When `--chat` is passed, the script runs llama.cpp with
`--conversation --single-turn --reasoning off` so llama.cpp applies the model's
chat template to the raw user prompt. The nixpkgs `llama-cli` currently keeps
some chat UI text in its raw stdout; the harness strips that from the
human-readable `-- generation --` section while preserving raw stdout in JSON.

## transformers Notes

`--transformers-model <hf-id-or-path>` attempts to import `transformers` and
`torch`. Those packages are intentionally not in the default flake yet because
they are heavy. When we add a dedicated correctness shell, prefer a separate
flake package set rather than making the normal Zig dev shell slow.

## What To Compare

Start with simple, deterministic smoke tests:

- tokenization IDs for short prompts
- first generated token for factual prompts
- short greedy/low-temperature continuations
- logit top-k for a single prefill position, once `llmtoy` exposes a debug mode

Avoid expecting long sampled generations to match exactly. Tiny differences in
dequantization, matvec order, or sampling filters can diverge quickly even when
both engines are broadly correct.

## Next Infrastructure Steps

- Add a `llmtoy logits` or `llmtoy inspect-logits` command for one-position
  top-k comparisons.
- Add tokenizer comparison mode that records token IDs from `llmtoy` and
  llama.cpp.
- Add checked fixture files under `tests/regression/` once the harness output is
  stable.
- Add a heavier optional Nix shell for `transformers`/`torch` comparisons.
