import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--subject", default="01", type=str, help="Subject to process",
)

parser.add_argument(
    "-l",
    "--level",
    default="1",
    type=str,
    help="Level to process",
    choices=["1", "4", "5"],
)
