import argparse
import logging

from cross_lingual_subnets.utils import set_device, set_seed

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training sentence embedding models")

    parser.add_argument(
        "--model",
        type=str,
        help="The model variant to train",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="The random seed for reproducibility"
        )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to use for training"
        )

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    args.device = args.device or set_device()

    # TODO: continue
