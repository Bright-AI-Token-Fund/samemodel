# samemodel

**Is your inference server still serving the same model?**

`samemodel` measures how far an LLM endpoint's output distribution moves when the
model is not supposed to change: a request that reuses the prompt cache, a request
batched together with others, a new engine version, another backend, another
quantisation. It talks to any OpenAI-compatible endpoint, has no dependencies and
needs no GPU of its own.

```text
$ samemodel selfcheck --base-url http://localhost:8080
samemodel selfcheck  http://localhost:8080  model=qwen2.5-0.5b-instruct-q4_k_m.gguf  24 prompts x 128 tokens  (unique conversations)

check                        same   argmax  drift50  drift99     KL99  forks  verdict
cache reuse                 1/12    97.87%   0.0888   0.4696   0.0116     11  drift
first request under load    1/12    97.86%   0.0752   0.3142   0.0073     11  drift
repeat under load           6/24    98.52%   0.0694   0.3689   0.0088     18  drift

cache reuse: DRIFT. 11 of 12 outputs changed; on 517 positions with an identical context the
  argmax agrees 97.87% of the time; the top-2 log-odds move by 0.0888 nat at the median and
  0.4696 nat at the 99th percentile.
  For scale: about 1.4 times the size of switching the weights from fp16 to Q8_0.
first request under load: DRIFT. 11 of 12 outputs changed; on 515 positions with an
  identical context the argmax agrees 97.86% of the time; the top-2 log-odds move by 0.0752
  nat at the median and 0.3142 nat at the 99th percentile.
  For scale: about the size of switching the weights from fp16 to Q8_0.
repeat under load: DRIFT. 18 of 24 outputs changed; on 1214 positions with an identical
  context the argmax agrees 98.52% of the time; the top-2 log-odds move by 0.0694 nat at the
  median and 0.3689 nat at the 99th percentile.
  For scale: about the size of switching the weights from fp16 to Q8_0.

Prompt tokens served from cache, as the endpoint reports it: 6% first request, 98% repeat.
Every conversation here starts differently. Run again with --shared-system to test requests
that share a system prompt, as most deployments do.
```

That is llama.cpp serving a Q4_K_M file on an Apple M4 CPU, at temperature 0. Same
weights, same server, same request. On the same machine a Q8_0 file is identical
bit for bit, and the Q4_K_M file is identical on a Xeon. Which case a deployment is
in cannot be read off the documentation, but it takes two minutes to measure.

## Try it in 30 seconds

No model needed. The built-in mock endpoint is a toy language model whose answers
can be made to depend on the serving path.

```bash
pipx install git+https://github.com/Bright-AI-Token-Fund/samemodel
samemodel mock --cache-drift 0.08 &          # a server whose prompt cache changes its answers
samemodel selfcheck --base-url http://127.0.0.1:8642
```

`uv tool install git+https://github.com/Bright-AI-Token-Fund/samemodel` works as
well. Python 3.10 or newer.

## Check your own server

Point it at the endpoint. It sends 24 short prompts three times each, greedy, 128
tokens, and takes a minute or two on a small model.

```bash
samemodel selfcheck --base-url http://localhost:8080             # llama-server, mlx_lm.server, vLLM, SGLang ...
samemodel selfcheck --base-url http://localhost:8080 --shared-system
samemodel selfcheck --base-url https://host/v1 --api-key $KEY --model my-model -n 60
```

Every prompt is sent three times: for the first time, again on its own, and again
at the same moment as three unrelated requests. A caching server can compute a
conversation for the first time only once, so every second prompt makes its first
request under load. That gives three comparisons:

| check | compares | tells you |
| --- | --- | --- |
| `cache reuse` | first request (alone) with the repeat | whether an answer computed from the prompt cache equals one computed from scratch |
| `first request under load` | first request (with company) with the repeat | the same, when the first computation shared the server with other requests; read it against the row above |
| `repeat under load` | the repeat with a third request sent in company | whether concurrent requests change an answer that is otherwise fully cached |

By default every conversation starts with a different system message, so a first
request has nothing to reuse. `--shared-system` gives all of them the same system
message instead, as most deployments do, and a first request then already reuses
the cached system prefix. The two modes answer different questions and can read
very differently: in our benchmark of the setup above, 22 of 30 answers change
between a first request and its repeat when conversations are unique, and 2 of 30
when they share a system prompt.

Requests carry no `seed`. Greedy decoding does not need one, and a seed can change
how a server schedules a request: `mlx_lm.server` does not batch requests that
have one, so its answers under load are exact with a seed and not without. If your
clients send a seed, pass `--seed N` to measure what they get.

The endpoint has to return `logprobs` with `top_logprobs` on
`/v1/chat/completions`. Without them the tool still compares the text and says that
it cannot size the difference.

Tested against llama.cpp (`llama-server`) and MLX (`mlx_lm.server`). vLLM and SGLang
expose the same fields and should work, but we have not run them yet; a
[measurement report](../../issues/new?template=measurement.yml) is welcome.

## Before and after a change

Record a run, change something (engine version, quantisation, flags, hardware),
record again, compare.

```bash
samemodel record --base-url http://localhost:8080 -o before.jsonl.gz
# ... upgrade, requantise, reconfigure ...
samemodel record --base-url http://localhost:8080 -o after.jsonl.gz
samemodel diff before.jsonl.gz after.jsonl.gz --forks 5
```

Or compare two live endpoints directly:

```bash
samemodel compare --a-base-url http://old:8080 --b-base-url http://new:8080
```

A recording is only as clean as the server's cache state. For a reference run,
either turn the prompt cache off for the recording (llama.cpp:
`--extra-body '{"cache_prompt": false}'`) or pass `--tag some-text`, which starts
every conversation with a unique system message so that nothing can be reused.

In CI, `--fail-on drift` (or `noise`, `different`) turns the verdict into an exit
status, and `--json out.json` writes every number.

## What we measured

One small model (Qwen2.5-0.5B-Instruct), llama.cpp b12244 and mlx-lm 0.32.2, each
weight file on a freshly started server:

| setup | weights | reusing the prompt cache | sharing the server with other requests |
| --- | --- | --- | --- |
| llama.cpp, M4 CPU | fp16 | noise: 5 of 30 answers changed, drift 0.0014 | noise: 3 of 60 answers changed, drift 0.0014 |
| llama.cpp, M4 CPU | q8_0 | identical | identical |
| llama.cpp, M4 CPU | q4_k_m | drift: 22 of 30 answers changed, drift 0.0838 | drift: 43 of 60 answers changed, drift 0.0658 |
| llama.cpp, M4 Metal | fp16 | noise: 1 of 30 answers changed, drift 0.0016 | noise: 3 of 60 answers changed, drift 0.0013 |
| llama.cpp, M4 Metal | q8_0 | noise: 1 of 30 answers changed, drift 0.0016 | noise: 4 of 60 answers changed, drift 0.0014 |
| llama.cpp, M4 Metal | q4_k_m | noise: 4 of 30 answers changed, drift 0.0026 | drift: 24 of 60 answers changed, drift 0.0146 |
| llama.cpp, Xeon CPU | fp16 | identical | identical |
| llama.cpp, Xeon CPU | q8_0 | identical | identical |
| llama.cpp, Xeon CPU | q4_k_m | identical | identical |
| MLX, M4 | bf16 | identical | drift: 45 of 60 answers changed, argmax 98.79% |
| MLX, M4 | 8bit | identical | noise: 8 of 60 answers changed, argmax 99.87% |
| MLX, M4 | 4bit | identical | noise: 10 of 60 answers changed, argmax 99.85% |

Drift is the median shift of the top-2 log-odds in nats, on 60 prompts, with every
conversation starting differently. MLX reports log-probabilities on a coarse grid,
so its rows show the argmax agreement instead. For scale, changing the weights on
the same machines:

| change of weights | median drift | argmax agreement |
| --- | --- | --- |
| fp16 replaced by Q8_0 (llama.cpp) | 0.05 to 0.08 | 98.1 to 98.8% |
| Q8_0 replaced by Q4_K_M (llama.cpp) | 0.32 to 0.35 | 93.1 to 93.9% |
| bf16 replaced by 8bit (MLX) | 0.08 | 98.6% |

Full method, hardware, versions, commands and raw data are in
[`benchmarks/`](benchmarks/README.md). The short version of what the numbers say:

- **A server that takes the same path twice gives the same answer twice.** Every
  difference we saw comes from a changed computation path, never from randomness.
- **Whether a path change matters depends on the combination** of quantisation,
  backend and CPU architecture, not on any one of them. The same Q4_K_M file drifts
  on the M4 CPU, far less on Metal, and not at all on the Xeon.
- **Where it matters, it can be as large as a quantisation step.** The engine did
  not get a different model file; it computed the same file differently.
- **A request field can decide the path.** `mlx_lm.server` batches only requests
  without a seed. Batched, its `bf16` build moves about as far as a swap to `8bit`
  weights would; with a seed nothing moves at all. The first version of this tool
  sent a seed and reported MLX as identical.

### Why, for one of these rows

The Q4_K_M row on the M4 CPU is traced to its causes in
[`benchmarks/root-cause/`](benchmarks/root-cause/README.md):

- **A batch of 32 or more tokens is computed by a different backend.** The macOS
  build includes Accelerate, which multiplies the activations as floats. Below 32
  rows the CPU kernel rounds them to 8 bits first. A prompt computed from scratch and
  the same prompt continued from the cache land on different sides of that line.
- **The single-row and the four-row Q6_K kernel for NEON round differently**, by
  about 4e-7. The 8-bit rounding of activations in the layers that follow turns that
  into steps a thousand times larger.

With the BLAS backend off and two lines of one kernel changed, the same file on the
same machine with the same batching is identical on all three checks. For a 1.5B
model of the same family, where the first cause does not apply, the two lines alone
are enough.

### Before comparing two runtimes, compare the files

llama.cpp serving the Q4_K_M file from Qwen's own GGUF repository and Ollama serving
`qwen2.5:0.5b` do not give one matching answer in 60, and the drift is 0.52 nat. Served
the same file, the two agree to rounding noise. What differs is the file: most GGUF
files in Qwen's repositories, the fp16 ones included, hold a rescaled copy of the
released weights, while Ollama's, bartowski's and lmstudio-community's are plain
conversions. In the two files we took apart, 0.5B and 7B at fp16, 1% of the weights
are clipped as well; on the 0.5B model that alone puts Qwen's fp16 file 0.25 nat away
from the checkpoint. The measurements, and a check that compares a file's norm weights
with the checkpoint without downloading either, are in
[`benchmarks/qwen-gguf/`](benchmarks/qwen-gguf/README.md).

## Reading the output

| column | meaning |
| --- | --- |
| `same` | prompts whose two outputs are token-for-token identical |
| `argmax` | on positions where both runs saw an identical context, how often they pick the same token |
| `drift50`, `drift99` | on those positions, how far the log-odds between the reference run's top two tokens moved, in nats (median and 99th percentile) |
| `KL99` | 99th percentile KL divergence over the tokens both runs list in their top-k |
| `forks` | prompts where the two outputs diverge |

| verdict | meaning |
| --- | --- |
| `identical` | same tokens and same log-probabilities everywhere |
| `noise` | rounding-level differences; outputs change only where two tokens were nearly tied |
| `drift` | a measurable shift of the distribution, up to roughly a small quantisation step; also a run that is mostly exact with a few positions well off |
| `different` | a larger shift: not the same model in practice |

The verdict is a reading aid. Its thresholds come from one set of measurements on
one small model, and the raw numbers are always printed next to it. The "for scale"
line compares the drift with reference points from the same measurements:

| reference point | median drift |
| --- | --- |
| fp16 weights on another serving path | 0.0015 nat |
| fp16 weights replaced by Q8_0 | 0.05 to 0.08 nat |
| Q8_0 weights replaced by Q4_K_M | 0.32 to 0.35 nat |

## How it works

Two greedy runs of the same prompt produce the same tokens up to the first position
where they disagree. Up to and including that position both runs scored exactly the
same context, so their top-k log-probabilities are directly comparable. After it the
contexts differ, and anything measured there mixes the endpoints' difference with
ordinary sensitivity to different input, so those positions are ignored.

This matters because the obvious metric saturates. "How many answers changed"
already reads 75 to 90% for a perturbation the size of fp16 to Q8_0 and 100% for one
a few times larger, since a single flipped near-tie changes everything after it.
Counting per position on identical contexts keeps the scale.

Drift is measured on log-odds rather than on agreement because agreement also
depends on the model: the same perturbation flips more tokens when the model's
distributions are flat.

## Limits

- It measures; it does not say whether a difference is a bug, and it does not find
  root causes.
- It sees what the endpoint reports. Engines that compute logits in half precision
  report log-probabilities on a grid (MLX: 1/128 to 1/16 nat), and drift below that
  reads as zero. The tool detects this, says so, and falls back to the argmax
  agreement for its verdict when the grid is too coarse.
- Load is best effort: the companion requests are sent at the same moment, but only
  the server decides what shares a batch, and it may treat requests differently
  depending on their fields (see the note on `seed` above).
- One live server cannot compute the same conversation from scratch twice, so "first
  request alone" and "first request under load" are measured on different halves of
  the prompts. A repeat also inherits whatever state the first request left in the
  cache, so an effect that is baked into the cached state cancels out. For a
  like-for-like comparison, record both sides on freshly started servers with
  `record --tag` and `record --tag --companions 3`, then `diff` them.
- Every run marks its conversations with a fresh random id, so two runs of the same
  check give slightly different numbers.
- The reference points and thresholds come from a 0.5B model. Expect the same
  ordering on larger models, not the same decimals.
- Chat completions only, no streaming, no tool calls.

## Related work

The phenomenon is known; this tool only makes it easy to measure on your own
deployment.

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  (Thinking Machines) explains batch invariance and why kernels lose it.
- vLLM documents a [batch-invariant mode](https://docs.vllm.ai/en/latest/features/batch_invariance/);
  SGLang has `--enable-deterministic-inference`.
- [Same Text, Different Prediction](https://arxiv.org/abs/2610.09111) gives the
  logit-margin argument for when a perturbation can flip a greedy token.
- llama.cpp [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) tracks
  greedy divergence under speculative decoding, cache reuse and batch size, and
  [#31504](https://github.com/ggml-org/llama.cpp/pull/31504) works on the x86 CPU
  kernels.

- [llamacpp-batch-invariance](https://github.com/addiplus/llamacpp-batch-invariance)
  checks whether concurrent requests to a llama.cpp server score the same as the same
  requests sent one at a time, and issues a pass or fail certificate per
  configuration. samemodel reports a distance instead, and also covers cache reuse,
  other engines and comparisons across builds.

Not to be confused with [SameModel](https://github.com/AkritiKeswani/samemodel), an
unrelated project that compares the providers of one model on OpenRouter for
tool-calling and formatting failures. That one asks what a provider does to a
request and its response; this one measures the numerical distance between two
servings of the same weights.

## How this project is made

samemodel is developed by an AI agent (Claude; see the commit trailers): the code,
the measurements and this README. A person sets the direction and signed off on
making the repository public; day-to-day changes are made by the agent.

That is why every number here is produced by a checked-in script from checked-in raw
data, with the hardware, the versions and the commands next to it. If a number looks
wrong, rerun it, and open an issue if it does not reproduce.

## License

Apache-2.0.
