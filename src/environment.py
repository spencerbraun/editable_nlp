import argparse
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"Debug: {args.debug}")

    print(utils.sailPreprocess(args.debug))
