# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline generation dump for Sharded-CP parity checks.

Run once with the flag off and once with ``--enable-sharded-context-parallel``,
then diff the two JSON files (the driver script does this automatically).
Greedy sampling; prompts cover short, batched-uneven, and long prefill.
"""

import argparse
import json


def build_prompts(long_repeat: int) -> list[str]:
    doc = (
        "The context parallel design splits token rows across ranks while "
        "keeping global cache slot semantics, so sparse attention can still "
        "address every token it selects. "
    )
    return [
        "The capital of France is",
        "def quicksort(arr):\n",
        "In one sentence, explain why the sky is blue:",
        # Long prompt: exercises prefill chunking and uneven CP splits.
        doc * long_repeat + "\nSummarize the paragraph above in ten words:",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--long-repeat", type=int, default=200)
    parser.add_argument("--enable-sharded-context-parallel", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        enforce_eager=True,
        enable_sharded_context_parallel=args.enable_sharded_context_parallel,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(build_prompts(args.long_repeat), sampling)

    records = []
    for output in outputs:
        completion = output.outputs[0]
        records.append(
            {
                "prompt_len": len(output.prompt_token_ids),
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "cumulative_logprob": completion.cumulative_logprob,
            }
        )
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} generations to {args.output}")


if __name__ == "__main__":
    main()
