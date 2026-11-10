# samemodel

**Is your inference server still serving the same model?**

`samemodel` measures how far an LLM endpoint's output distribution moves when the
model is not supposed to change: a request that reuses the prompt cache, a request
batched together with others, a new engine version, another backend, another
quantisation. It talks to any OpenAI-compatible endpoint, has no dependencies and
needs no GPU of its own.

```text
$ samemodel selfcheck --base-url http://localhost:8080

check             same   argmax  drift50  drift99     KL99  forks  verdict
<!-- EXAMPLE_TABLE -->

<!-- EXAMPLE_TEXT -->
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

Point it at the endpoint. It sends 20 short prompts three times each, greedy, 128
tokens, and takes a minute or two on a small model.

```bash
samemodel selfcheck --base-url http://localhost:8080             # llama-server, vLLM, SGLang, mlx_lm.server ...
samemodel selfcheck --base-url https://host/v1 --api-key $KEY --model my-model -n 60
```

For each prompt it compares three requests:

| request | what it is | compared with |
| --- | --- | --- |
| scratch | the first time the server sees this conversation | |
| cached | the identical request again, so the server can reuse its prompt cache | scratch, as **cache reuse** |
| busy | again, sent at the same moment as three unrelated requests | cached, as **co-batched** |

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

<!-- MEASURED -->

Full method, hardware, versions, commands and raw data are in
[`benchmarks/`](benchmarks/README.md). The short version of what the numbers say:

- **Repeating the same request under the same conditions is always identical.**
  Every difference comes from a changed computation path, never from randomness.
- **Whether a path change matters depends on the combination** of quantisation,
  backend and CPU architecture, not on any one of them.
- **Where it matters, it can be as large as a quantisation step.** The engine did
  not get a different model file; it computed the same file differently.

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
| `drift` | a measurable shift of the distribution, up to roughly a small quantisation step |
| `different` | a large shift, or a small median with a heavy tail: not the same model in practice |

The verdict is a reading aid. Its thresholds come from one set of measurements on
one small model, and the raw numbers are always printed next to it. The "for scale"
line compares the drift with reference points from the same measurements:

| reference point | median drift |
| --- | --- |
| fp16 weights, different batch shape | 0.0015 nat |
| fp16 weights replaced by Q8_0 | 0.05 to 0.08 nat |
| Q8_0 weights replaced by Q4_K_M | 0.32 to 0.35 nat |

## How it works

Two greedy runs of the same prompt produce the same tokens up to the first position
where they disagree. Up to and including that position both runs scored exactly the
same context, so their top-k log-probabilities are directly comparable. After it the
contexts differ, and anything measured there mixes the endpoints' difference with
ordinary sensitivity to different input, so those positions are ignored.

This matters because the obvious metric saturates. "How many answers changed"
already reads 85% for a perturbation the size of fp16 to Q8_0 and 100% for one a few
times larger, since a single flipped near-tie changes everything after it. Counting
per position on identical contexts keeps the scale.

Drift is measured on log-odds rather than on agreement because agreement also
depends on the model: the same perturbation flips more tokens when the model's
distributions are flat.

## Limits

- It measures; it does not say whether a difference is a bug, and it does not find
  root causes.
- It sees what the endpoint reports. Engines that compute logits in half precision
  report log-probabilities on a coarse grid (MLX: 1/128 to 1/16 nat), and drift below
  that reads as zero. The tool detects this and says so.
- Co-batching is best effort: the companions are sent at the same moment, but only
  the server decides what shares a batch.
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

## License

Apache-2.0.
