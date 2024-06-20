import argparse
import multiprocessing
import os
import pickle
from collections import defaultdict
import shutil

# from pybloom_live import ScalableBloomFilter
from tqdm import tqdm

# from line_profiler import LineProfiler

num_workers = 9
bf_init_capacity = 100000
bf_error_rate = 0.001
CONTENT_FIELD_NAME = "raw_content"
DOC_ID_FIELD_NAME = "doc_id"
HASH_FIELD_NAME = "hash"


def process_dir(input_dir, output_path):
    lsh_dict = defaultdict(str)
    with open(output_path, "w") as fw:
        for file_name in os.listdir(input_dir):
            print(f"begin process {file_name}")
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "rb") as f:
                doc_list = pickle.load(f)
            # output_path = os.path.join(output_dir, file_name)
            for doc in doc_list:
                key = doc[DOC_ID_FIELD_NAME]
                H = doc[HASH_FIELD_NAME]
                cand = lsh_dict.get(H, "None")
                if cand != "None":
                    fw.write(f"{key} :: {cand}\n")
                else:
                    lsh_dict[H] = key
            print(f"finish process file {file_name}")


def process_partition(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    pool = multiprocessing.Pool(processes=num_workers)
    for bucket_dir_name in os.listdir(input_dir):
        bucket_dir_path = os.path.join(input_dir, bucket_dir_name)
        output_path = os.path.join(output_dir, f"dup_paris_{bucket_dir_name}.txt")
        pool.apply_async(process_dir, (bucket_dir_path, output_path))
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    process_partition(input_dir, output_dir)
