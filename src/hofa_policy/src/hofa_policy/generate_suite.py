"""Generate a deterministic obstacle-free HOFA teacher trajectory suite."""
import argparse
import json

from .trajectory_suite import generate_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=5)
    parser.add_argument("--test", type=int, default=5)
    args = parser.parse_args()
    if min(args.train, args.val, args.test) < 0:
        parser.error("split counts must be nonnegative")
    print(json.dumps(generate_suite(
        args.output, args.seed, args.train, args.val, args.test), indent=2))


if __name__ == "__main__":
    main()
