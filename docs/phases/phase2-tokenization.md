# Phase 2 — Tokenization

## What is tokenization?

LLMs don't operate on raw bytes or characters — they operate on *tokens*, which are sub-word units learned from a training corpus. The two main schemes:

**BPE (Byte Pair Encoding)** — used by Qwen3, GPT-2, LLaMA, most recent models:
1. Start with an alphabet of all 256 byte values.
2. Repeatedly merge the most frequent adjacent pair into a new token.
3. Record each merge in order — the merge list is the trained vocabulary extension.
4. At inference time, encode text by applying merges in the same order.

**Unigram / SentencePiece** — used by older LLaMA, Gemma2:
- Maintains a probability for each token; encoding finds the sequence that maximises likelihood.
- Not used by our target models (Gemma4 uses BPE despite being SentencePiece-lineage).

Both Qwen3 and Gemma4 ship BPE in their GGUF files (`tokenizer.ggml.merges` array).

## GGUF tokenizer keys

Keys observed across our two target models:

| key | type | description |
|-----|------|-------------|
| `tokenizer.ggml.model` | string | `"gpt2"` (Qwen3) or `"gemma4"` (Gemma4) |
| `tokenizer.ggml.tokens` | `[string]` | vocab strings in ID order |
| `tokenizer.ggml.merges` | `[string]` | BPE merge rules, each `"left right"` |
| `tokenizer.ggml.token_type` | `[int32]` | normal=1, control=3, byte=6, … |
| `tokenizer.ggml.scores` | `[float32]` | SentencePiece scores (Gemma4 only) |
| `tokenizer.ggml.bos_token_id` | uint32 | beginning-of-sequence ID |
| `tokenizer.ggml.eos_token_id` | uint32 | end-of-sequence ID |
| `tokenizer.ggml.add_bos_token` | bool | prepend BOS at encode time |

## bytes_to_unicode

GPT-2 BPE tokens are byte-level, but token strings in the vocabulary are human-readable unicode text, not raw bytes. The trick: define a bijection between the 256 byte values and 256 unicode codepoints that avoids control characters and whitespace.

Three ranges of bytes already have clean, printable unicode codepoints and map to themselves:
- `0x21..0x7E` (printable ASCII `!..~`, 94 bytes)
- `0xA1..0xAC` (Latin-1 `¡..¬`, 12 bytes)
- `0xAE..0xFF` (Latin-1 `®..ÿ`, 82 bytes)

The remaining 68 bytes (0x00–0x20, 0x7F, 0x80–0xA0, 0xAD) map to codepoints 0x100–0x143 in iteration order. Most notably:

- space (0x20) → U+0120 `Ġ`  — visible in every GPT-2 family vocabulary as the word-leading space marker
- byte 0x00 → U+0100 `Ā`
- byte 0x7F → U+0121 `ġ`

This mapping is a fixed table, computed at comptime in `src/tokenizer/bpe.zig`.

```zig
// byte → unicode codepoint (bijection, 256 entries)
const BYTE_TO_CP: [256]u21 = blk: { ... };

// codepoint → byte (reverse, 324 entries covering all output codepoints)
const CP_TO_BYTE: [324]u8 = blk: { ... };
```

`CP_TO_BYTE` is indexed by codepoint value. `cpIsByteEncoded(cp)` guards the decode path for codepoints that fall outside the 256 valid outputs (e.g. literal unicode in special tokens like `<|im_end|>`).

## BPE encoding algorithm

Given a word (a pre-tokenized piece):

1. **Byte-encode**: convert each byte through `BYTE_TO_CP`, produce a list of UTF-8 strings — one per byte.
2. **Merge loop**: find the adjacent pair `(symbols[i], symbols[i+1])` whose concatenation with a space separator (`"left right"`) has the lowest rank in `merge_rank`. Merge it. Repeat until no pair exists in `merge_rank`.
3. **Vocab lookup**: map each final symbol string to its token ID via `token_to_id`.

Merge rank = index in `tokenizer.ggml.merges` (lower index → applied first). The merge key format exactly matches how merges are stored in GGUF: `"left right"` with a space separator.

The algorithm is O(n² · m) per word (n symbols, m merge attempts per pass), which is fine for educational use. Production implementations (llama.cpp, tiktoken) use a priority queue for O(n log n).

## Pre-tokenization (simplification)

GPT-2 family tokenizers never merge across word boundaries. Real models use a regex to split text into pieces before BPE — e.g. Qwen3's `"qwen35"` pre-tokenizer uses a tiktoken-compatible Unicode-aware pattern.

Our phase 2 uses a simple whitespace split: words are space-separated, and every word after the first gets a leading `Ġ` (space byte mapped through `BYTE_TO_CP`). This is correct for basic ASCII prose but will differ from the real tokenizer on:
- punctuation attached to words (`"hello,"` splits differently)
- numbers (`"42"` may be split differently by tiktoken's digit rules)
- non-ASCII text

**Concrete artifact — Gemma4 splits `jumps` into `j` + `umps`:**

```
  245237  Ġ          ← space is its own token
  236804  j
   12909  umps
```

This happens because Gemma4's merge table was trained with the space *included* in the token — `Ġjumps` is a single high-frequency unit in the training corpus. Presented with bare `jumps` (no leading space), the merge `j` + `umps` → `jumps` ranks lower than the merges that build `umps` first (`u`+`m`→`um`, `um`+`p`→`ump`, `ump`+`s`→`umps`), so `j` is left stranded.

The real Gemma4 pre-tokenizer feeds `Ġjumps` as the atomic piece to BPE and gets a single token. Our implementation emits Ġ as a standalone token and strips it from the following piece, producing the artifact above. Qwen3 doesn't exhibit this because it folds the space *into* the word before BPE (`Ġjumps` is the input, not `Ġ` + `jumps`), which is exactly the GPT-2 convention its merge table was trained on.

This is enough to demonstrate the BPE algorithm correctly. A full regex pre-tokenizer is a separate phase.

## What we built

`src/tokenizer/vocab.zig` — `Vocab` struct + `fromGguf`:
- Extracts `tokens`, `bos/eos_token_id`, `add_bos` from parsed GGUF metadata
- Builds `token_to_id` (string→u32 hashmap) and `merge_rank` (pair→u32 hashmap)
- Borrows token/merge strings from the GgufData arena — no copies

`src/tokenizer/bpe.zig` — encode + decode:
- Comptime `BYTE_TO_CP` / `CP_TO_BYTE` tables
- `encode(text, vocab, allocator) ![]u32`
- `decode(ids, vocab, allocator) ![]u8`

`src/gguf/reader.zig` — added `metaArray` and `metaBool` accessors to `GgufData` and `GgufReader`.

## Results on target models

**Qwen3.6 35B (gpt2 tokenizer, vocab=248320, merges=247587)**:
```
input:   "The quick brown fox jumps over the lazy dog"
ids:     { 760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388 }
decoded: "The quick brown fox jumps over the lazy dog"

tokens:
   760  The
  3841  Ġquick
 13477  Ġbrown
 37550  Ġfox
 33075  Ġjumps
   888  Ġover
   279  Ġthe
 15217  Ġlazy
  5388  Ġdog
```

**Gemma4 26B (gemma4 tokenizer, vocab=262144, merges=514906, add_bos=true)**:
```
input:   "hello world"
ids:     { 2, 23391, 245237, 12392 }
decoded: "<bos>hello world"

tokens:
       2  <bos>      ← add_bos=true, prepended automatically
   23391  hello
  245237  Ġ          ← space is a standalone token in Gemma4
   12392  world
```

Note the different conventions: Qwen3 merges the leading space into the following word (`Ġworld`), while Gemma4 keeps space as a standalone token (`Ġ` + `world`). Both decode correctly back to the original string.

## Next

Phase 3: naive CPU inference — matmul, RMSNorm, softmax, SiLU, single-head attention + FFN forward pass.
