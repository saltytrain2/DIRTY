import os

def main():
    with open("files.txt", "w") as dataset_file:
        for file in os.listdir("./output/bins"):
            dataset_file.write(file + "\n")
    pass


if __name__ == "__main__":
    main()
