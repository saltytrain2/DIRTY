import os
import sys

def main(argv):
    if not len(argv):
        raise ValueError("Please specify the path to the directory to generate filenames")

    types_set = set()
    bins_set = set()

    for file in os.listdir(f"{argv[0]}/bins"):
        bins_set.add(file[:file.index("_")])

    for file in os.listdir(f"{argv[0]}/types"):
        types_set.add(file[:file.index("_")])
    
    bin_files = list(set.intersection(types_set, bins_set))
    print(len(bin_files))

    with open("files.txt", "w") as dataset_file:
        for file in bin_files:
            dataset_file.write(f"{file}_{file}.jsonl.gz\n")
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
