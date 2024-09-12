"""
Usage:
    infer.py [options] INPUT_JSON

Options:
    -h --help                  Show this screen.
"""

from utils.ghidra_function import CollectedFunction

from docopt import docopt
import json
import gzip

def main(args):
    json_dict = json.load(open(args["INPUT_JSON"], "r"))
    #print(json_dict)

    cf = CollectedFunction.from_json(json_dict)
    print(cf)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
