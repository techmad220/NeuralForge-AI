from __future__ import annotations

import argparse
import ast
from typing import List
import json
from pathlib import Path

import numpy as np

from .core import DynamicSlicingAI
from .chunked_model import create_random_model, load_metadata
from .datasets import SyntheticVectorDataset


def _parse_hidden(xs: List[str]) -> List[int]:
    return [int(x) for x in xs]


def cmd_init_model(args: argparse.Namespace) -> None:
    hidden = _parse_hidden(args.hidden)
    create_random_model(args.outdir, args.d_in, hidden, args.d_out, seed=args.seed)
    meta = json.dumps(load_metadata(args.outdir), indent=2)
    print(f"Created model at {args.outdir}\n{meta}")


def cmd_infer(args: argparse.Namespace) -> None:
    x = np.array(ast.literal_eval(args.input), dtype=np.float32)
    ai = DynamicSlicingAI(args.model_dir, slicing=args.slicing, continual=args.continual)
    out = ai.run_inference(x, return_logits=args.logits)
    print(out.tolist())


def cmd_list_plugins(_: argparse.Namespace) -> None:
    print(json.dumps(DynamicSlicingAI.available_plugins(), indent=2))


def cmd_learn_from(args: argparse.Namespace) -> None:
    ai = DynamicSlicingAI(args.student, slicing=args.slicing, continual=args.continual)
    meta = load_metadata(args.student)
    ds = SyntheticVectorDataset(d_in=meta["d_in"], n=args.samples, seed=args.seed)
    ai.learn_from_other_ai(
        other_ai_model_dir=args.teacher,
        dataset=ds,
        lambda_l2=args.lmbda,
        temperature=args.temp,
    )
    print("Student head updated via KD-head.")


def main() -> None:
    p = argparse.ArgumentParser("dynai")
    sub = p.add_subparsers(required=True)

    p_init = sub.add_parser("init-model", help="Create a random chunked model on disk.")
    p_init.add_argument("--outdir", required=True)
    p_init.add_argument("--d-in", type=int, required=True)
    p_init.add_argument("--hidden", nargs="+", required=True, help="e.g. --hidden 32 32")
    p_init.add_argument("--d-out", type=int, required=True)
    p_init.add_argument("--seed", type=int, default=1)
    p_init.set_defaults(func=cmd_init_model)

    p_infer = sub.add_parser("infer", help="Run sliced inference.")
    p_infer.add_argument("--model-dir", required=True)
    p_infer.add_argument("--input", required=True, help='Python list, e.g. "[0.1,0.2,...]"')
    p_infer.add_argument("--logits", action="store_true", help="Return logits instead of probs.")
    p_infer.add_argument("--slicing", default=None)
    p_infer.add_argument("--continual", default=None)
    p_infer.set_defaults(func=cmd_infer)

    p_list = sub.add_parser("list-plugins", help="Show available slicing/continual plugins.")
    p_list.set_defaults(func=cmd_list_plugins)

    p_kd = sub.add_parser("learn-from", help="KD-head continual learning (student <- teacher).")
    p_kd.add_argument("--student", required=True)
    p_kd.add_argument("--teacher", required=True)
    p_kd.add_argument("--samples", type=int, default=256)
    p_kd.add_argument("--seed", type=int, default=42)
    p_kd.add_argument("--lmbda", type=float, default=0.05)
    p_kd.add_argument("--temp", type=float, default=2.0)
    p_kd.add_argument("--slicing", default=None)
    p_kd.add_argument("--continual", default=None)
    p_kd.set_defaults(func=cmd_learn_from)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()