from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--latent", default=None, help="path to checkpoint to restore latent")

    opt = parser.parse_args()
    return opt
