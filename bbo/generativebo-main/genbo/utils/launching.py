import argparse
import logging
import os
from contextlib import contextmanager

from genbo.utility import utilities_registry


def add_base_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m", "--num-test", type=int, default=1000, help="Number of test examples"
    )
    parser.add_argument(
        "-n",
        "--num-train",
        type=int,
        default=100,
        help="Number of training examples (must be less than `num_test`)",
    )
    parser.add_argument(
        "-u",
        "--utility",
        choices=utilities_registry.keys(),
        default="PI",
        help="Expected utility function",
    )
    parser.add_argument(
        "--noise-sd", default=0.2, type=float, help="Noise standard deviation"
    )
    parser.add_argument(
        "--threshold", default=0.5, type=float, help="Threshold percentile for utility"
    )
    parser.add_argument(
        "--num-hidden",
        default=128,
        type=int,
        help="Number of hidden units for conditional flow",
    )
    parser.add_argument(
        "--num-layers",
        default=3,
        type=int,
        help="Number of layers for conditional flow",
    )
    parser.add_argument(
        "--num-transforms", default=3, type=int, help="Number of transforms in flow"
    )
    parser.add_argument("--flow-class", type=str, default="NSF", help="Zuko flow class")
    parser.add_argument(
        "--num-steps", default=1000, type=int, help="Number of training steps"
    )
    parser.add_argument(
        "--weight-decay", default=0, type=float, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--lr", default=1e-2, type=float, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--use-scheduler", action="store_true", help="Use learning rate scheduler"
    )
    parser.add_argument(
        "--train-seed",
        default=None,
        type=int,
        help="Optional different RNG seed to apply before training",
    )
    parser.add_argument(
        "--lengthscale",
        default=1.0,
        type=float,
        help="Lengthscale for objective function generation",
    )


def setup_logging(logfile: str = ""):
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )


@contextmanager
def change_dir(destination: str):
    original_dir = os.getcwd()
    os.chdir(destination)
    try:
        yield
    finally:
        os.chdir(original_dir)
