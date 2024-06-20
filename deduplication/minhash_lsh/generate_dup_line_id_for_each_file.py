import argparse
import pickle
import os
from collections import defaultdict
import shutil
import tqdm


def generate_duplicates(input_file, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print("Processing duplicates!!!")
    # load pickled components and other artifacts
    with open(input_file, "rb") as fin:
        components, n_components, reversed_mapper = pickle.load(fin)

    duplicates = defaultdict(set)
    n_duplicate_docs = 0
    for component in tqdm.tqdm(components):
        for j in range(1, len(component)):
            doc = reversed_mapper[component[j]]
            file_name, line_idx = doc.split("@")
            duplicates[file_name].add(int(line_idx))
            n_duplicate_docs += 1

    print("number of duplicate documents that will be removed:", n_duplicate_docs)

    for file_name, line_idx_set in duplicates.items():
        output_path = os.path.join(output_dir, file_name + ".pkl")
        with open(output_path, "wb") as fout:
            pickle.dump(line_idx_set, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    generate_duplicates(args.input_file, args.output_dir)
