# Most official Qwen GGUF files are not a plain conversion of the released weights

We wanted to know how far apart Ollama and llama.cpp are when they serve the same
model. For Qwen2.5-0.5B-Instruct the two runtimes turned out to agree to rounding
noise. The two files did not: the Q4_K_M file in Qwen's own GGUF repository and the
Q4_K_M file that `ollama pull qwen2.5:0.5b` downloads give different answers to every
one of 60 prompts at temperature 0. This page finds out why.

1. **The runtime is not the cause.** The same file gives the same answers in
   llama.cpp b12244 and in Ollama 0.40.16, up to rounding noise: 50 or 51 of 60
   answers are identical and the top-2 log-odds move by 0.002 nat at the median.
2. **The files hold different weights.** Not one quantised tensor has the same bytes
   in the two Q4_K_M files, and neither do the normalisation weights, which are stored
   unquantised in both.
3. **Qwen's files for this model hold a rescaled and clipped copy of the checkpoint.** In every
   block the norm weights differ from the released checkpoint, by up to 6.9. The
   matrices that read a norm output are the checkpoint's, multiplied per input channel
   by the inverse factor. That part leaves the model unchanged. On top of it, 1.06% of
   the matrix weights are smaller in magnitude than the rescaled checkpoint value, and
   in every group of 128 input channels they sit at one common ceiling. That is what
   AWQ's scale and clip steps produce. Ollama's files are a plain conversion: their
   norm weights equal the checkpoint's exactly.
4. **The clip changes the model.** The fp16 file in Qwen's repository is 0.25 nat away
   from a plain fp16 conversion of the checkpoint, about three quarters of the step
   from Q8_0 to Q4_K_M, and none of the 60 answers match. When we restore the clipped
   weights and keep the rescale, 55 of 60 answers match and the distance is 0.003 nat.
5. **It is not specific to this model.** The norm weights can be compared without
   downloading a file. They differ from the checkpoint in 17 of the 21 official files
   we checked: every Qwen2.5 file from 0.5B to 32B, the Coder models, QwQ-32B, and the
   Q4_K_M files of Qwen3. They are identical in the Q8_0 files of Qwen3, in Qwen's
   Qwen2-7B file, and in every build by Ollama, bartowski and lmstudio-community. The
   one larger file we downloaded in full, Qwen2.5-7B at fp16, is clipped the same way:
   1.07% of its matrix weights.

Qwen's documentation describes the procedure as a way to
[improve the quality of quantised models](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html#quantizing-the-gguf-with-awq-scale).
The model cards of the GGUF repositories do not mention it (we read six of them on
2026-11-11). Some files hint at it in their metadata: 5 of the 17 rescaled files
have `AWQ`, `Awq` or `AWInt4` in their `general.name`, and
[others have noticed](#related-work) such labels. The other 12, among them every
Qwen2.5 Instruct file from 0.5B to 32B, carry no such label and are rescaled all the
same.

Nothing here says the files are worse. It says they are a different set of weights
from every other build of the same model, including at "fp16".

## Setup

- Apple M4, 16 GB, macOS. llama.cpp b12244 (official `macos-arm64` release) and Ollama
  0.40.16, both on Metal with default settings. Ollama ran every file with the
  `llama-server` it ships.
- Checkpoint: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
  `model.safetensors`, bf16.
- Qwen's files: `qwen2.5-0.5b-instruct-q4_k_m.gguf` and `qwen2.5-0.5b-instruct-fp16.gguf` from
  [`Qwen/Qwen2.5-0.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).
- Ollama's files: the model layers of `qwen2.5:0.5b` (Q4_K_M) and
  `qwen2.5:0.5b-instruct-fp16`.
- 60 prompts, 128 tokens each, greedy decoding with `top_logprobs`, no system message,
  one freshly started server per file. Two recordings are compared at the positions
  where both saw the same context. Drift is the median shift of the top-2 log-odds in
  nats; see the main [README](../../README.md#how-it-works).

The reference for "the released model" is Ollama's fp16 file. It is a plain
conversion: every matrix in it equals the checkpoint to within 5e-8
([`data/rescale_check.ollama-fp16.txt`](data/rescale_check.ollama-fp16.txt)).

## The runtime is not the cause

Each file was recorded once through llama.cpp and once through Ollama:

| file | identical answers | argmax agreement | median drift |
| --- | --- | --- | --- |
| plain fp16 conversion | 51 of 60 | 99.94% | 0.0016 |
| Qwen's Q4_K_M | 50 of 60 | 99.96% | 0.0018 |
| Ollama's Q4_K_M | 51 of 60 | 99.97% | 0.0016 |

That is the size of the rounding noise between two batch shapes on this backend (see
the [benchmarks](../README.md)). Importing Qwen's GGUF file into Ollama with a one-line
Modelfile gave the same number of prompt tokens as llama.cpp and the agreement in the
second row: Ollama's generated Modelfile shows `TEMPLATE {{ .Prompt }}`, but the chat
template embedded in the file is what gets applied.

## The files are not the same

All in llama.cpp, so that only the file changes:

| from | to | identical answers | argmax agreement | median drift |
| --- | --- | --- | --- | --- |
| plain fp16 conversion | Qwen's fp16 | 0 of 60 | 95.76% | 0.255 |
| plain fp16 conversion | Qwen's fp16, clip undone | 55 of 60 | 99.93% | 0.0026 |
| plain fp16 conversion | Qwen's Q4_K_M | 0 of 60 | 93.05% | 0.412 |
| plain fp16 conversion | Ollama's Q4_K_M | 0 of 60 | 90.81% | 0.412 |
| Qwen's fp16 | Qwen's Q4_K_M | 0 of 60 | 93.67% | 0.334 |
| Qwen's fp16 | Ollama's Q4_K_M | 0 of 60 | 90.83% | 0.417 |
| Qwen's Q4_K_M | Ollama's Q4_K_M | 0 of 60 | 89.05% | 0.516 |

For scale, on the same machine a swap from fp16 to Q8_0 is 0.05 to 0.08 and a swap
from Q8_0 to Q4_K_M is 0.32 to 0.35.

Three things to read off this table:

- Qwen's fp16 file is not the checkpoint at full precision. It is 0.255 away from it.
- The two Q4_K_M files are further from each other (0.516) than either is from the
  checkpoint (0.412).
- Measured against the checkpoint, the two Q4_K_M files are equally far at the median.
  Qwen's agrees with the checkpoint's top token more often, 93.05% against 90.81%.
  This is one small model, and the two files also differ in how they store the
  embedding (below), so it is not a ranking of the two builds.

## What is in the files

[`tensor_diff.py`](tensor_diff.py) hashes every tensor of the two Q4_K_M files
([output](data/tensor_diff.q4_k_m.txt)):

| tensors | Qwen's file | Ollama's file | bytes |
| --- | --- | --- | --- |
| 132 | Q5_0 | Q5_0 | different |
| 72 | F32 | F32 | same (the biases) |
| 49 | F32 | F32 | different (the norm weights) |
| 12 | Q8_0 | Q8_0 | different |
| 12 | Q6_K | Q6_K | different |
| 12 | Q4_K | Q4_K | different |
| 1 | Q5_0 | Q8_0 | different (`token_embd.weight`) |
| 1 | Q8_0 | absent | `output.weight` |

The type of each tensor is the same in both files, with one exception. Qwen's file
stores the token embedding twice, once at Q5_0 for the input and once at Q8_0 for the
output; Ollama's stores it once at Q8_0 and uses it for both. In Qwen's fp16 file both
copies equal the checkpoint's embedding, so this only matters after quantisation.

The norm weights are F32 in both files, so they can be compared exactly. Ollama's
equal the checkpoint in all 49 tensors. Qwen's equal it in one, the final
`output_norm`, and differ in the other 48 by up to 6.86, on values that go up to 12.

[`rescale_check.py`](rescale_check.py) compares Qwen's fp16 file with the checkpoint,
matrix by matrix, over all 24 blocks ([output](data/rescale_check.qwen-gguf-fp16.txt)).
For the layers that read a norm output it takes the factor
`s = checkpoint norm / file norm` per channel and tests whether
`file matrix = checkpoint matrix × s`. A scale that is not tied to a norm, such as the one
between the rows of `ffn_up` and the columns of `ffn_down`, is estimated as the median
ratio of the elements. Relative differences are Frobenius norms: the median over the
blocks in the second column, the largest over the blocks in the third.

| layer | file vs checkpoint | file vs rescaled checkpoint | weights the rescale does not explain | of those, smaller in magnitude |
| --- | --- | --- | --- | --- |
| `attn_q` | 0.367 | 0.0003 | 0 | |
| `attn_k` | 0.370 | 0.0003 | 0 | |
| `attn_v` | 0.383 | 0.093 | 28,488 (1.03%) | 100% |
| `ffn_gate` | 0.310 | 0.038 | 1,016,795 (0.97%) | 100% |
| `ffn_up` | 0.271 | 0.042 | 1,226,979 (1.17%) | 100% |
| `attn_output` | 0.053 | 0.088 | 333,984 (1.73%) | 100% |
| `ffn_down` | 0.340 | 0.044 | 1,202,626 (1.15%) | 100% |

The query and key projections are the checkpoint's, rescaled, to within fp16
rounding. In the other five kinds of matrix, about 1% of the weights differ from the
rescaled checkpoint by more than 1%, and every one of them is smaller in magnitude. In
total that is 3,808,872 of 357,826,560 matrix weights, 1.06%.

The reduced weights are not scattered. Split each row into groups of 128 consecutive
input channels: in every one of the 972,057 groups that hold two or more reduced
weights, those weights share a single magnitude, and it is the largest in the group. A
ceiling per output row and per group of 128 input channels is how
[AutoAWQ](https://github.com/casper-hansen/AutoAWQ) clips, with its default group
size. The rescale between a norm and the layers that read it is how it scales.

## The clip is what moves the output

`rescale_check.py --unclip` rewrites a copy of Qwen's fp16 file so that every matrix is
exactly the rescaled checkpoint. The norm weights and the rescale stay as they are in
Qwen's file; only the 1.06% of reduced weights change by more than rounding.

| file | identical answers with the plain conversion | median drift |
| --- | --- | --- |
| Qwen's fp16 | 0 of 60 | 0.255 |
| Qwen's fp16 with the clip undone | 55 of 60 | 0.0026 |

So the rescale preserves the model, as it should, and the whole distance between
Qwen's fp16 file and the checkpoint comes from the clipped weights.

## Which files are affected

Norm weights are unquantised and small, so a GGUF file can be checked against its
checkpoint without downloading either: [`remote_norm_check.py`](remote_norm_check.py)
reads the GGUF header and the norm tensors with HTTP range requests, about 10 MB per
file, and compares them with the same tensors of the safetensors checkpoint. For files
split into parts it reads the first part. `identical` means equal in every element.

| file | `general.name` in the file | norm tensors compared | identical | different | largest difference |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf` | qwen2.5-0.5b-instruct | 49 | 1 | 48 | 6.86 |
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf` | qwen2.5-0.5b-instruct | 49 | 1 | 48 | 6.86 |
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-fp16.gguf` | qwen2.5-0.5b-instruct | 49 | 1 | 48 | 6.86 |
| `Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf` | qwen2.5-1.5b-instruct | 57 | 1 | 56 | 4.68 |
| `Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf` | qwen2.5-3b-instruct | 73 | 1 | 72 | 26.5 |
| `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | qwen2.5-7b-instruct | 47 | 0 | 47 | 4.81 |
| `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf` | qwen2.5-7b-instruct | 27 | 0 | 27 | 4.81 |
| `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-fp16-00001-of-00004.gguf` | qwen2.5-7b-instruct | 13 | 0 | 13 | 4.81 |
| `Qwen/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf` | qwen2.5-14b-instruct | 43 | 0 | 43 | 3.69 |
| `Qwen/Qwen2.5-32B-Instruct-GGUF/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf` | qwen2.5-32b-instruct | 23 | 0 | 23 | 4.01 |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | Qwen2.5 Coder 1.5B Instruct GGUF | 57 | 1 | 56 | 5.48 |
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf` | Qwen2.5 Coder 7B Instruct GGUF | 47 | 0 | 47 | 4.39 |
| `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/qwen2.5-coder-32b-instruct-q4_k_m-00001-of-00003.gguf` | Qwen2.5 Coder 32B Instruct AWQ | 51 | 0 | 51 | 4.67 |
| `Qwen/QwQ-32B-GGUF/qwq-32b-q4_k_m.gguf` | Tmp_Quanted_Model AWInt4 Groupsize128 | 129 | 1 | 128 | 4.19 |
| `Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` | Qwen3 4B Instruct Awq | 73 | 1 | 72 | 10.4 |
| `Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` | Qwen3 8B Awq Compatible Instruct | 73 | 1 | 72 | 11.3 |
| `Qwen/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf` | Qwen3 14B Awq Compatible Instruct | 81 | 1 | 80 | 11.6 |
| `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` | Qwen3 0.6B Instruct | 57 | 57 | 0 | 0 |
| `Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf` | Qwen3 1.7B Instruct | 57 | 55 (+2 within rounding) | 0 | 2.24e-08 |
| `Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf` | Qwen3 4B Instruct | 73 | 73 | 0 | 0 |
| `Qwen/Qwen2-7B-Instruct-GGUF/qwen2-7b-instruct-q4_k_m.gguf` | qwen2-7b-instruct | 57 | 57 | 0 | 0 |
| `bartowski/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | Qwen2.5 0.5B Instruct | 49 | 49 | 0 | 0 |
| `bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Qwen2.5 7B Instruct | 57 | 57 | 0 | 0 |
| `bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` | Qwen2.5 Coder 7B Instruct | 57 | 57 | 0 | 0 |
| `lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | Qwen2.5 0.5B Instruct | 49 | 49 | 0 | 0 |
| `lmstudio-community/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Qwen2.5 7B Instruct | 57 | 57 | 0 | 0 |

Ollama's library files were read from its registry the same way:

| file | `general.name` in the file | norm tensors compared | identical | different | largest difference |
| --- | --- | --- | --- | --- | --- |
| `library/qwen2.5:0.5b (sha256:c5396e06af29)` | Qwen2.5 0.5B Instruct | 49 | 49 | 0 | 0 |
| `library/qwen2.5:7b (sha256:2bada8a74506)` | Qwen2.5 7B Instruct | 57 | 57 | 0 | 0 |
| `library/qwen2.5-coder:7b (sha256:60e05f210007)` | Qwen2.5 Coder 7B Instruct | 57 | 57 | 0 | 0 |

Three things stand out:

- Every Qwen2.5 file is affected, at every level we looked at, including Q8_0 and fp16.
- In the Qwen3 repositories the Q4_K_M files are rescaled and the Q8_0 files are not.
  The two levels of one repository are quantisations of two different sets of float
  weights.
- The third-party builds and Ollama's are plain conversions, so they agree with the
  checkpoint and differ from Qwen's files.
- The name in the metadata is not a reliable sign. Five of the rescaled files point
  to AWQ in their `general.name` (QwQ-32B's reads `Tmp_Quanted_Model AWInt4
  Groupsize128`); twelve do not. No file whose norm weights match the checkpoint does.

This check shows the rescale only. To see whether a larger file is clipped as well we
downloaded one in full: the four parts of `qwen2.5-7b-instruct-fp16` and the checkpoint
`Qwen/Qwen2.5-7B-Instruct`, 15 GB each, and ran `rescale_check.py` over all 28 blocks
([output](data/rescale_check.qwen2.5-7b-fp16.txt)):

| layer | file vs rescaled checkpoint | weights the rescale does not explain | of those, smaller in magnitude |
| --- | --- | --- | --- |
| `attn_q` | 0.0003 | 6,728 (0.002%) | |
| `attn_k` | 0.0003 | 1,472 (0.003%) | |
| `attn_v` | 0.048 | 503,155 (0.98%) | 100% |
| `ffn_gate` | 0.035 | 19,255,503 (1.01%) | 100% |
| `ffn_up` | 0.048 | 22,358,320 (1.18%) | 100% |
| `attn_output` | 0.051 | 5,096,692 (1.42%) | 100% |
| `ffn_down` | 0.077 | 22,575,427 (1.19%) | 100% |

It is the same picture as the 0.5B file. The query and key projections are the
checkpoint's, rescaled (we did not examine the few thousand weights that differ
there). In the other five kinds of matrix 69,789,097 of 6,525,288,448 weights are
smaller than the rescaled checkpoint value, 1.07%, and in 99.93% of the groups of 128
input channels that hold two or more of them they share one ceiling.

How far the clip moves the output of the 7B model we have not measured. That takes a
full-precision run of both files. For the other files the norm check is all we have.

## What this means in practice

- **An fp16 GGUF file from Qwen's repository is not a full-precision reference for the
  checkpoint.** If you measure how much a quantisation loses by comparing against it,
  you are comparing against the clipped model. On the 0.5B model that reference is
  already 0.25 nat away from the checkpoint.
- **"Qwen2.5 at Q4_K_M" is at least two different models.** A result obtained with a
  file from Qwen's repository and one obtained with Ollama's, bartowski's or
  lmstudio-community's file are results for different weights. That includes
  comparisons of runtimes that take each runtime's default download.
- **If answers differ between Ollama and llama.cpp, compare the files first.** With
  the same file the two agreed to rounding noise here.
- **To check a file you already have,** compare its norm weights with the checkpoint.
  It takes seconds and it is exact.

## Related work

- Qwen's documentation has a section on
  [quantising the GGUF with AWQ scale](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html#quantizing-the-gguf-with-awq-scale),
  presented as an option for people who quantise the models themselves.
- In a [discussion on bartowski's Qwen2.5-Coder-32B repository](https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/discussions/3)
  (February 2025) a user notes that the models in the official repository "are
  quantized using AWQ", and bartowski answers that his are not.
- [Broken on Arrival](https://arxiv.org/abs/2609.05881) (Patodiya, 2026) and the
  [Ollama issue](https://github.com/ollama/ollama/issues/18252) it grew out of show
  that four low-bit `qwen2.5-coder:3b` tags of Ollama's library match a plain rebuild
  from the checkpoint in all 434 tensors, and note that Qwen's own Coder-3B files are
  labelled `Instruct-AWQ` and store a separate output head.

What this page adds: a comparison of the weights themselves instead of the labels
(the rescale, the clip and its structure); that files without any AWQ label are
rescaled too, the fp16 ones included; a measurement of what the clip does to the
output, with an intervention that isolates it; and a check that works on a remote
file.

## Reproduce

```bash
pip install numpy          # the scripts in this directory need it; samemodel itself does not
cd benchmarks/qwen-gguf

# which tensors differ between two files
python tensor_diff.py qwen2.5-0.5b-instruct-q4_k_m.gguf ollama-qwen2.5-0.5b.gguf

# rescale and clip, against the checkpoint (for split files: comma-separated parts and shards)
python rescale_check.py qwen2.5-0.5b-instruct-fp16.gguf model.safetensors

# undo the clip in a copy
cp qwen2.5-0.5b-instruct-fp16.gguf unclipped.gguf
python rescale_check.py qwen2.5-0.5b-instruct-fp16.gguf model.safetensors --unclip unclipped.gguf

# record every setup and print the distance tables (see the variables at the top of run.sh)
./run.sh && ./distances.sh

# check files on Hugging Face or in the Ollama registry without downloading them
python scan.py targets.txt
python remote_norm_check.py GGUF_URL SAFETENSORS_URL [SAFETENSORS_URL ...]
python ollama_registry_check.py library/qwen2.5:7b Qwen/Qwen2.5-7B-Instruct
```

An Ollama model layer is a GGUF file. `ollama show --modelfile qwen2.5:0.5b` prints
its path.

## Limits

- The distances are from one model of 0.5B parameters, one machine and one backend.
  Small models react more to a perturbation than large ones. Do not carry the numbers
  over to the 7B or 32B files: for 7B we showed that the weights are clipped, not what
  that does to the output.
- For the files other than the 0.5B and the 7B fp16 ones we compared the norm weights
  only. That shows they were rescaled. It does not show that they were clipped.
- For split files only the first part was read: 23 to 57 norm tensors, not all of them.
- That the files were made with AutoAWQ is an inference from the structure of the
  weights, from Qwen's documentation and from the names in some files' metadata. We
  found no statement from Qwen about how the published files were made.
- The checkpoint was read from mlx-community's bf16 copy for the matrices. A sample of
  seven tensors was compared with Qwen's own file and is identical
  ([`data/checkpoint_copy_check.txt`](data/checkpoint_copy_check.txt)); the norm
  weights were compared with Qwen's own file directly.
- 60 prompts of 128 tokens. A median drift is stable at that size; a difference of two
  points in argmax agreement between two files is not something to build on.
- This analysis was done by an AI model (see the commit trailers). We have not
  contacted the Qwen team about it.

## Data

- [`data/runs/`](data/runs): the eight recordings. `checkpoint-f16` is the plain fp16
  conversion, `qwen-gguf-*` are Qwen's files, `ollama-q4_k_m` is Ollama's; a
  `.via-ollama` suffix means the file was served by Ollama instead of llama.cpp.
- [`data/distances.txt`](data/distances.txt): output of `distances.sh`.
- `data/tensor_diff.q4_k_m.txt`, `data/rescale_check.*.txt`,
  `data/checkpoint_copy_check.txt`: outputs of the scripts of the same names. The 7B
  check ran on a Xeon server, CPU only, and took nine minutes.
- [`data/scan.jsonl`](data/scan.jsonl) and [`data/scan.ollama.jsonl`](data/scan.ollama.jsonl):
  one line per file checked remotely, 2026-11-11. [`targets.txt`](targets.txt) lists the
  repositories; `scan_table.py` prints the tables above.
