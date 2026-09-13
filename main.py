#!/usr/bin/env python3
"""Main entry point for DWRF V4."""
import sys
import argparse
from pathlib import Path

from experiments.runner import DWRFRunner
from config.config import load_config


def main():
    parser = argparse.ArgumentParser(description="DWRF V4 Experiment Runner")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output:
        config.output.dir = args.output

    runner = DWRFRunner(config)
    results = runner.run()
    print("Experiment completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())