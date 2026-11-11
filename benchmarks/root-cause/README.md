# Why llama.cpp output at temperature 0 depends on the batch on an Apple M4 CPU

The [benchmarks](../README.md) show llama.cpp b12244 serving a Q4_K_M file on an
Apple M4 CPU with outputs that depend on the serving path: reusing the prompt cache
or sharing the server with other requests moves the distribution by 0.07 to 0.08 nat,
as much as replacing fp16 weights with Q8_0. A Q8_0 file on the same machine is
identical bit for bit. This page finds out why, for that one setup.

There are two independent causes and one amplifier.

1. **A batch of 32 or more tokens is computed by a different backend.** The macOS
   release includes a BLAS backend (Accelerate). It takes over a matrix
   multiplication once the batch has 32 rows, for every weight type that has no
   repacked CPU kernel. It multiplies the weights by the activations as floats. Below
   32 rows the CPU backend first rounds the activations to 8 bits. The two results
   differ by about 0.5% per multiplication. This is what `cache reuse` measures: a
   prompt computed from scratch goes through BLAS, the same prompt continued from the
   cache does not.
2. **The single-row and the four-row Q6_K kernel round differently.** On NEON,
   `ggml_gemv_q6_K_8x4_q8_K` multiplies and then adds; `ggml_gemm_q6_K_8x4_q8_K` uses
   a multiply-accumulate. The results differ in the last bit, about 4e-7 relative.
   This is what the `under load` checks measure: a token computed alone goes through
   the first kernel, a token computed with three others through the second.
3. **Rounding the activations to 8 bits amplifies the second cause.** A perturbation
   of 4e-7 changes nothing until it pushes one activation across a rounding boundary.
   Then the output of that multiplication moves by about 3e-4, a thousand times more.
   With float weights the same perturbation stays at 4e-7.

Remove both causes and the same file, on the same machine, with default batching, is
identical on all three checks. For a second model, whose tensors are all of types with
a repacked kernel, the first cause does not apply, and changing two lines of the Q6_K
kernel is enough.

## The experiment

Everything below uses the official `qwen2.5-0.5b-instruct-q4_k_m.gguf` on the M4 CPU
(`--device none -ngl 0 -t 4 -c 16384 -np 4`), 60 prompts, unique conversations, unless
a row says otherwise. Each cell shows the verdict, how many outputs stayed the same,
the argmax agreement, and the median / 99th percentile drift in nats.

| BLAS backend | Q6_K kernels | cache reuse | first request under load | repeat under load |
| --- | --- | --- | --- | --- |
| on | as released | **drift**<br>8/30 same, argmax 98.81%<br>drift 0.0838 / 0.4202 | **drift**<br>4/30 same, argmax 98.37%<br>drift 0.0710 / 0.3929 | **drift**<br>17/60 same, argmax 98.87%<br>drift 0.0658 / 0.3583 |
| on | patched | **drift**<br>9/30 same, argmax 98.86%<br>drift 0.0790 / 0.4250 | **drift**<br>21/30 same, argmax 99.65%<br>drift 0.0000 / 0.2262 | **identical**<br>60/60 same |
| off | as released | **noise**<br>26/30 same, argmax 99.88%<br>drift 0.0000 / 0.0848 | **drift**<br>5/30 same, argmax 98.45%<br>drift 0.0680 / 0.3335 | **drift**<br>11/60 same, argmax 98.54%<br>drift 0.0656 / 0.3396 |
| off | patched | **identical**<br>30/30 same | **identical**<br>30/30 same | **identical**<br>60/60 same |

The first row is the official release binary. The other three are builds of the same
commit from source, with the release's instruction-set baseline (`-DGGML_NATIVE=OFF`):
"patched" applies [`q6k-8x4-gemv.patch`](q6k-8x4-gemv.patch), "off" configures with
`-DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF`.

The same two causes can be removed without building anything, by changing the file or
the batch size on the release binary:

| what changed | cache reuse | first request under load | repeat under load |
| --- | --- | --- | --- |
| the file quantised again from fp16 with this build's `llama-quantize` (control) | **drift**<br>3/30 same, argmax 98.25%<br>drift 0.0817 / 0.4298 | **drift**<br>6/30 same, argmax 98.15%<br>drift 0.0714 / 0.4128 | **drift**<br>10/60 same, argmax 98.42%<br>drift 0.0715 / 0.4369 |
| no Q6_K tensors (`--tensor-type ffn_down=q4_k`) | **drift**<br>5/30 same, argmax 98.27%<br>drift 0.0882 / 0.3958 | **drift**<br>21/30 same, argmax 99.68%<br>drift 0.0000 / 0.2373 | **identical**<br>60/60 same |
| more Q6_K tensors (`--tensor-type ffn_down=q6_k`) | **drift**<br>5/30 same, argmax 98.34%<br>drift 0.0826 / 0.4109 | **drift**<br>5/30 same, argmax 98.24%<br>drift 0.0751 / 0.3769 | **drift**<br>11/60 same, argmax 98.44%<br>drift 0.0688 / 0.3815 |
| official file, one token per batch (`-ub 1`) | **identical**<br>30/30 same | **identical**<br>30/30 same | **identical**<br>60/60 same |

A file without Q6_K tensors behaves like the patched build. And for that file the
batch size at which the output starts to differ from one token per batch is exactly
32 (one slot, prompt cache off, 30 prompts of 36 to 58 tokens, 48 new tokens each):

| `-ub` | same as `-ub 1` | argmax agreement | median drift | verdict |
| --- | --- | --- | --- | --- |
| 2 | 30/30 | 100.00% | 0.0000 | identical |
| 3 | 30/30 | 100.00% | 0.0000 | identical |
| 4 | 30/30 | 100.00% | 0.0000 | identical |
| 5 | 30/30 | 100.00% | 0.0000 | identical |
| 8 | 30/30 | 100.00% | 0.0000 | identical |
| 16 | 30/30 | 100.00% | 0.0000 | identical |
| 32 | 9/30 | 97.63% | 0.0968 | drift |
| 64 | 10/30 | 97.81% | 0.0971 | drift |
| 512 | 10/30 | 97.81% | 0.0971 | drift |

## Cause 1: 32 rows and the BLAS backend

What the file contains matters here. `gguf_types.py` lists it:

```
Q5_0   133 tensors   attn_q, attn_k, attn_output, ffn_gate, ffn_up, half of attn_v, token_embd
Q8_0    13 tensors   the other half of attn_v, output
Q6_K    12 tensors   ffn_down
Q4_K    12 tensors   ffn_down
```

The model is 896 wide, and K-quants need a row length that is a multiple of 256, so
`llama-quantize` fell back to Q5_0 and Q8_0 for every tensor whose input is the hidden
state. Only `ffn_down`, with 4864 inputs, is a K-quant.

In the source of this commit:

- The CPU backend keeps Q4_0, Q8_0, Q4_K, Q5_K, Q6_K and IQ4_NL weights in a "repack"
  buffer on this CPU. That buffer type is
  [not a host buffer](https://github.com/ggml-org/llama.cpp/blob/d5bbcd1f31dba754f716a26c3a4784502a6b9ede/ggml/src/ggml-cpu/repack.cpp#L5253).
- The BLAS backend
  [only works on host buffers](https://github.com/ggml-org/llama.cpp/blob/d5bbcd1f31dba754f716a26c3a4784502a6b9ede/ggml/src/ggml-blas/ggml-blas.cpp#L452-L453),
  and it
  [takes a matrix multiplication](https://github.com/ggml-org/llama.cpp/blob/d5bbcd1f31dba754f716a26c3a4784502a6b9ede/ggml/src/ggml-blas/ggml-blas.cpp#L419-L432)
  when both dimensions of the weight and the number of rows are at least
  `min_batch = 32`, for any weight type that can be converted to float. It converts
  the weights and calls `cblas_sgemm` on the float activations.

So the Q5_0 tensors of this file are multiplied in two ways. With fewer than 32 rows,
the CPU kernel rounds each activation row to 8 bits and takes integer dot products.
With 32 rows or more, Accelerate multiplies floats. `amplification.cpp` measures the
distance between the two for one multiplication (its last column): 5.3e-3 to 5.4e-3 of
the output's RMS for Q5_0 and Q8_0, 6.7e-3 to 7.8e-3 for the K-quants. The Q8_0 file
has none of this because all of its tensors are repacked, and BLAS never sees them.

Weight types without a repacked kernel on this CPU, as `batch_invariance` reports
them: Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
F16, BF16, F32. We measured Q5_0. For the others this is a reading of the code, not
a measurement.

`first request under load` is less affected than `cache reuse` (21/30 against 9/30
with the patch) although both compute a prompt from scratch. We did not trace this. A
likely reason is that the server splits a batch that holds several conversations into
smaller pieces, so the last prompt token often lands in one with fewer than 32 rows.

## Cause 2: the Q6_K kernel pair

For repacked weights the CPU backend
[sends rows in groups of four to a `gemm` kernel and the remaining rows to a `gemv` kernel](https://github.com/ggml-org/llama.cpp/blob/d5bbcd1f31dba754f716a26c3a4784502a6b9ede/ggml/src/ggml-cpu/repack.cpp#L4637-L4648).
A token computed alone is one row and goes to `gemv`. The same token in a batch of
four goes to `gemm`. The two must agree bit for bit for the batch size not to matter.

`batch_invariance.cpp` checks that directly: it multiplies up to 96 different
activation vectors in one call and compares every column with the same vector
multiplied alone. On the release libraries, for the shape of `ffn_down`:

```
type   buffer   rows  max |batch - alone|    columns that differ
q4_K   repack      4  0.000e+00              0 of 4
q4_K   repack     96  0.000e+00              0 of 96
q6_K   repack      3  0.000e+00              0 of 3
q6_K   repack      4  4.375e-07              4 of 4
q6_K   repack      5  4.375e-07              4 of 5
q6_K   repack     96  7.194e-07              96 of 96
q5_0   plain      96  0.000e+00              0 of 96
q8_0   repack     96  0.000e+00              0 of 96
```

Only Q6_K differs, and only for the rows that went through `gemm`. In the source, the
last step of the two kernels:

```c
// ggml_gemv_q6_K_8x4_q8_K, arch/arm/repack.cpp line 1554
float32x4_t w_0123 = vmulq_f32(vcvtq_f32_s32(acc[0]), sb_scale_0);
acc_f32[0] = vaddq_f32(acc_f32[0], w_0123);

// ggml_gemm_q6_K_8x4_q8_K, arch/arm/repack.cpp line 4964
acc_f32[idx0] = vmlaq_f32(acc_f32[idx0], vcvtq_f32_s32(acc_0123), sbd_scale_0123[row]);
```

A multiply followed by an add rounds twice; the multiply-accumulate is compiled to a
fused operation that rounds once. The Q4_K kernels use `vfmaq_f32` in both places and
agree exactly. [`q6k-8x4-gemv.patch`](q6k-8x4-gemv.patch) changes the two lines of the
single-row kernel to the form the four-row kernel uses. With it, `batch_invariance`
reports zero for Q6_K at every batch size up to 96
(`data/batch_invariance.q6k-patch.4864x896.txt`), and `repeat under load` in the table
above becomes identical.

The release binary is built without `i8mm`, so it uses the `8x4` kernels. A build
with `-mcpu=native` on an M2 or later selects the `8x8` kernels instead. Their source
shows the same pattern (`vaddq_f32` of a product in `gemv`, `vmlaq_f32` in `gemm`); we
did not build or measure that variant.

## The amplifier: activations rounded to 8 bits

A difference of 4e-7 in twelve tensors sounds harmless, and the drift it causes is
0.07 nat. `amplification.cpp` shows why: perturb the input of one multiplication by a
relative `eps` and look at how far the output moves.

```
type    eps=1e-07     eps=4e-07     eps=1e-06     eps=1e-05     eps=1e-04     activations
f32     2.01e-07      3.82e-07      7.18e-07      6.33e-06      6.32e-05      kept as float
f16     1.17e-07      2.79e-07      6.38e-07      6.31e-06      6.32e-05      kept as float
q8_0    0.00e+00      0.00e+00      0.00e+00      3.36e-04      1.00e-03      rounded to 8 bits
q5_0    0.00e+00      0.00e+00      0.00e+00      3.35e-04      1.00e-03      rounded to 8 bits
q6_K    1.26e-07      2.91e-07      6.13e-07      3.17e-04      1.46e-03      rounded to 8 bits
q4_K    2.50e-07      4.11e-07      6.91e-07      3.16e-04      1.46e-03      rounded to 8 bits
```

(The output has one more column, used in cause 1 above.) With float activations the
output moves in proportion. With rounded activations
nothing moves until one of the 4864 values crosses a rounding boundary, and then the
output jumps by about 3e-4 whatever the size of the perturbation. (For other seeds
the first jump comes at 1e-6 or at 1e-4; the file in `data/` has three.) A model has
some 170 such multiplications per token, so a perturbation that starts at 4e-7 does
not stay there.

This also explains why the Q8_0 file is exact. It has the amplifier but no source of
perturbation: all of its tensors are repacked, BLAS never sees them, and its kernel
agrees with itself at every batch size. The fp16 file is the opposite case. Its kernel
differs by 2e-6 to 4e-6 between one row and several, and its tensors are open to BLAS,
but nothing rounds its activations to 8 bits; its drift in the main benchmark is
0.0014 nat. We did not analyse that file further.

## Does it carry over to another model

`qwen2.5-1.5b-instruct-q4_k_m.gguf` is 1536 wide, a multiple of 256, so nothing falls
back: it has 169 Q4_K and 29 Q6_K tensors, and all of them are repacked on this CPU.
By the reasoning above the BLAS backend should play no part, and the kernel pair
should be the only cause. On the M4 CPU, 24 prompts, BLAS on in both rows:

| build | cache reuse | first request under load | repeat under load |
| --- | --- | --- | --- |
| release | **drift**<br>11/12 same, argmax 99.92%<br>drift 0.0000 / 0.4039 | **drift**<br>1/12 same, argmax 97.76%<br>drift 0.1383 / 0.5283 | **drift**<br>2/24 same, argmax 97.73%<br>drift 0.1328 / 0.6071 |
| same commit with the two-line patch | **identical**<br>12/12 same | **identical**<br>12/12 same | **identical**<br>24/24 same |

Cache reuse is nearly exact on the release, as expected without BLAS in the path: 11
of 12 answers are the same. Under load the drift is about twice that of the smaller
model, 0.13 to 0.14 nat. The patch alone removes all of it.

## What this means in practice

- On the release binary for macOS, a prompt of 32 or more new tokens and the same
  prompt continued from the cache are computed differently for every tensor that has
  no repacked kernel. How much of a file that is depends on the quantisation and on
  the model's width: here it was 133 of 170 weight tensors, because 896 is not a
  multiple of 256.
- Files with Q6_K tensors (Q4_K_M, Q5_K_M and Q6_K all have them) give a token a
  different result depending on whether it is computed alone or in a group of four,
  on the NEON build.
- `-ub 1` removes both effects on the release binary, at the price of processing the
  prompt one token at a time. `-ub 16` or less removes the first one.
- Neither effect is an error in the sense of a wrong result. Each path computes the
  quantised model to within its own rounding. What is lost is that two requests for
  the same thing get the same answer.

## Reproduce

Model-level rows, with the scripts of this repository (`SERVER_FLAGS` is passed to
`llama-server`):

```bash
# one token per batch
SERVER_FLAGS="--device none -ngl 0 -ub 1" QUANTS=q4_k_m bash benchmarks/llamacpp.sh <llama-server> <models dir> out/ub1

# files without and with more Q6_K tensors
llama-quantize --tensor-type ffn_down=q4_k qwen2.5-0.5b-instruct-fp16.gguf no-q6k.gguf Q4_K_M
llama-quantize --tensor-type ffn_down=q6_k qwen2.5-0.5b-instruct-fp16.gguf all-q6k.gguf Q4_K_M
python benchmarks/root-cause/gguf_types.py no-q6k.gguf

# batch size sweep: record with each -ub, then compare with -ub 1
samemodel record --base-url http://127.0.0.1:8080 -n 30 --max-tokens 48 \
    --extra-body '{"cache_prompt": false}' -o ub32.jsonl.gz
samemodel diff ub1.jsonl.gz ub32.jsonl.gz
```

Builds from source, at the commit of the release:

```bash
git fetch --depth 1 origin d5bbcd1f31dba754f716a26c3a4784502a6b9ede && git checkout FETCH_HEAD
git apply q6k-8x4-gemv.patch                                   # "patched"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_METAL=OFF \
      -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF                    # "off"; leave these two out for "on"
cmake --build build --target llama-server
```

Unit tests, against the libraries of a release and the headers of the same commit:

```bash
clang++ -std=c++17 -O2 -I <llama.cpp>/ggml/include benchmarks/root-cause/batch_invariance.cpp \
    -L <release dir> -lggml -lggml-base -lggml-cpu -Wl,-rpath,<release dir> -o batch_invariance
./batch_invariance 4864 896      # inputs, outputs of the weight matrix
```

## Limits

- Two models of one family, one llama.cpp build, one machine. The mechanism is in the
  code and should hold more widely; the numbers will not.
- `batch_invariance` exercises the CPU backend alone, without the BLAS backend and
  without the rest of the graph. On the M4 its one finding was confirmed by
  intervention: patching the kernel pair it flags removes the drift it predicts.
- On the Xeon the same unit test reports differences between batched and single-row
  results for several types (`data/xeon/batch_invariance.xeon.*.txt`), yet the server
  there gives identical output at `-ub 1` and `-ub 512` for all three weight files (8
  prompts, 32 tokens, `data/xeon/`) and is identical in every selfcheck. We have not
  explained that difference. Do not read the unit test's x86 output as a finding.
- The Linux release has no BLAS backend, which is consistent with cause 1 being
  absent there.
- llama.cpp tracks batch invariance in
  [#25618](https://github.com/ggml-org/llama.cpp/issues/25618), and
  [#31504](https://github.com/ggml-org/llama.cpp/pull/31504) proposes fixes for x86
  kernels. We found no report of the NEON Q6_K pair or of the 32-row switch.
- This analysis was done by an AI model (see the commit trailers). llama.cpp asks
  that bug reports are written by people, so we have not filed one. Everything needed
  to check the result is in this directory.

## Data

```
data/release.selfcheck.unique.json               official file, release binary (copy of ../data/llamacpp-m4-cpu/)
data/q6k-patch.selfcheck.unique.json             BLAS on, kernels patched
data/no-blas.selfcheck.unique.json               BLAS off, kernels as released
data/no-blas-q6k-patch.selfcheck.unique.json     BLAS off, kernels patched
data/release-requantised.selfcheck.unique.json   release binary, file quantised again (control)
data/release-no-q6k.selfcheck.unique.json        release binary, file without Q6_K tensors
data/release-all-q6k.selfcheck.unique.json       release binary, every ffn_down tensor Q6_K
data/release-ub1.selfcheck.unique.json           release binary, official file, -ub 1
data/1_5b-release.selfcheck.unique.json          the 1.5B file, release binary
data/1_5b-q6k-patch.selfcheck.unique.json        the 1.5B file, BLAS on, kernels patched
data/ub-sweep-no-q6k/ub<N>.jsonl.gz              recordings of the batch size sweep
data/batch_invariance.release-b12244.*.txt       unit test output, M4, release libraries
data/batch_invariance.q6k-patch.4864x896.txt     unit test output, M4, patched build
data/amplification.release-b12244.txt            unit test output, M4, three seeds
data/xeon/                                       unit test output and -ub recordings on the Xeon
```

The tables above come from these files:

```bash
python benchmarks/summarize.py selfcheck "x=benchmarks/root-cause/data" \
    --weights release q6k-patch no-blas no-blas-q6k-patch
```
