#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 给数据打上标签
import os
import time
from tqdm import tqdm
import glob
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import time
from joblib import Parallel, delayed
import json
import shutil
from deduplication.minhash_lsh.generate_minhash import process_dir
from deduplication.minhash_lsh.generate_dup_pairs import process_partition
from deduplication.minhash_lsh.generate_connected_components import (
    generate_connected_components_mp,
)
from deduplication.minhash_lsh.generate_dup_line_id_for_each_file import (
    generate_duplicates,
)
from deduplication.minhash_lsh.remove_dup import remove_dup_in_dir
import gzip
import pickle
import math
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return int(obj)
        if isinstance(obj, np.integer):
            return float(obj)
        if isinstance(obj, np.floating):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_jsonl(data, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False, cls=NumpyEncoder)
            f.write("\n")


def move_code_file():
    github_dir = "/data0/datasets/codeparrot/github_tag"
    starcode_dir = "/data0/datasets/starcodder/starcodder_tag"
    stackv2 = "/data0/datasets/bigcode/stackv2/stackv2_tag"
    save_dir = "/data0/datasets/dep_codess"
    names = list(set(os.listdir(github_dir) + os.listdir(starcode_dir))+ os.listdir(stackv2))
    for name in names:
        print("name:", name)
        output_dir = os.path.join(save_dir, name, "source")
        if os.path.exists(output_dir):
            print("exists:", output_dir)
            # continue
        else:
            os.makedirs(output_dir)

        if name in os.listdir(stackv2):
            sub_dir = os.path.join(stackv2, name)
            if os.path.isdir(sub_dir):
                for file in os.listdir(sub_dir):
                    src = os.path.join(sub_dir, file)
                    dst = os.path.join(output_dir, file)
                    print("dsl file:",dst)
                    shutil.copy(src, dst)

        if name in os.listdir(github_dir):
            sub_dir = os.path.join(github_dir, name)
            if os.path.isdir(sub_dir):
                for file in os.listdir(sub_dir):
                    src = os.path.join(sub_dir, file)
                    dst = os.path.join(output_dir, file)
                    shutil.copy(src, dst)
        if name in os.listdir(starcode_dir):
            sub_dir = os.path.join(starcode_dir, name)
            if os.path.isdir(sub_dir):
                for file in os.listdir(sub_dir):
                    src = os.path.join(sub_dir, file)
                    dst = os.path.join(output_dir, file)
                    shutil.copy(src, dst)


# step1
def calculate_hash(data_type="code",data_dir="/data0/datasets/dep_codess", is_input_dir=False,num_worker=128):
    if is_input_dir:
        input_dir = os.path.join(data_dir, "source")
        output_dir = os.path.join(data_dir, "hash")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        process_dir(input_dir, output_dir, num_worker=num_worker,data_type=data_type)
    else:
        for name in os.listdir(data_dir):
            input_dir = os.path.join(data_dir, name, "source")
            output_dir = os.path.join(data_dir, name, "hash")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            # else:
            #     print("exits:",output_dir)
            #     continue
            process_dir(input_dir, output_dir, num_worker=num_worker,data_type=data_type)


# step2
def calculate_dup_pairs(data_dir="/data0/datasets/dep_codess", is_input_dir=False):
    if is_input_dir:
        input_dir = os.path.join(data_dir, "hash")
        output_dir = os.path.join(data_dir, "dup_pairs")
        process_partition(input_dir, output_dir)
    else:
        for name in os.listdir(data_dir):
            input_dir = os.path.join(data_dir, name, "hash")
            output_dir = os.path.join(data_dir, name, "dup_pairs")
            process_partition(input_dir, output_dir)


# step3
def calculate_connected_components(
    data_dir="/data0/datasets/dep_codess", is_input_dir=False
):
    if is_input_dir:
        input_dir = os.path.join(data_dir, "dup_pairs")
        output_file = os.path.join(data_dir, "connected_components.json")
        generate_connected_components_mp(input_dir, output_file, num_workers=32)
    else:
        for name in os.listdir(data_dir):
            input_dir = os.path.join(data_dir, name, "dup_pairs")
            output_file = os.path.join(data_dir, name, "connected_components.json")
            generate_connected_components_mp(input_dir, output_file, num_workers=32)


# step4
def calculate_dup_line_each_file(
    data_dir="/data0/datasets/dep_codess", is_input_dir=False
):
    if is_input_dir:
        input_file = os.path.join(data_dir, "connected_components.json")
        output_dir = os.path.join(data_dir, "dup_line_each_file")
        generate_duplicates(input_file, output_dir)
    else:
        for name in os.listdir(data_dir):
            input_file = os.path.join(data_dir, name, "connected_components.json")
            output_dir = os.path.join(data_dir, name, "dup_line_each_file")
            generate_duplicates(input_file, output_dir)


# step5
def remove_dup(data_dir="/data0/datasets/dep_codess", is_input_dir=False,num_workers=32):
    if is_input_dir:
        input_dir = os.path.join(data_dir, "source")
        dup_line_id_dir = os.path.join(data_dir, "dup_line_each_file")
        output_dir = os.path.join(data_dir, "dedup")
        remove_dup_in_dir(input_dir, dup_line_id_dir, output_dir, num_workers=num_workers)
    else:
        for name in os.listdir(data_dir):
            print("processing:", name)
            input_dir = os.path.join(data_dir, name, "source")
            dup_line_id_dir = os.path.join(data_dir, name, "dup_line_each_file")
            output_dir = os.path.join(data_dir, name, "dedup")
            remove_dup_in_dir(input_dir, dup_line_id_dir, output_dir, num_workers=num_workers)


def remove_dup_dataset(name):
    data_dir = "/data0/datasets/dep_codes"
    # step1 hash
    # print("start hash....")
    # input_dir = os.path.join(data_dir, name, "source")
    # output_dir = os.path.join(data_dir, name, "hash")
    # if not os.path.exists(output_dir):
    #     process_dir(input_dir, output_dir, num_worker=128)

    # step2 dup_pairs
    print("start dup_pairs....")
    input_dir = os.path.join(data_dir, name, "hash")
    output_dir = os.path.join(data_dir, name, "dup_pairs")
    # process_partition(input_dir, output_dir)
    if not os.path.exists(output_dir):
        process_partition(input_dir, output_dir)

    # step3 connected_components
    print("start connected_components....")
    input_dir = os.path.join(data_dir, name, "dup_pairs")
    output_file = os.path.join(data_dir, name, "connected_components.json")
    if not os.path.exists(output_file):
        generate_connected_components_mp(input_dir, output_file, num_workers=12)

    # step4 dup_line_each_file
    print("start dup_line_each_file....")
    input_file = os.path.join(data_dir, name, "connected_components.json")
    output_dir = os.path.join(data_dir, name, "dup_line_each_file")
    if not os.path.exists(output_dir):
        generate_duplicates(input_file, output_dir)

    # step5 remove_dup
    print("start remove_dup....")
    input_dir = os.path.join(data_dir, name, "source")
    dup_line_id_dir = os.path.join(data_dir, name, "dup_line_each_file")
    output_dir = os.path.join(data_dir, name, "dedup")
    if not os.path.exists(output_dir):
        remove_dup_in_dir(input_dir, dup_line_id_dir, output_dir, num_workers=12)


def split_map_cc(data_dir="/data0/datasets/MAP-CC/map-cc/source"):
    files = os.listdir(data_dir)
    output_dir = "/data0/datasets/MAP-CC/map-cc/source_trans"
    files = ["general_zh_cc.jsonl"]
    for file in files:
        datas = []
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for index, line in tqdm(enumerate(f)):
                datas.append(json.loads(line))
                if len(datas)>10000000:
                    output_file_name = (file.split(".")[0] + "_" + str(len(os.listdir(output_dir))) + ".jsonl")
                    output_path = os.path.join(output_dir, output_file_name)
                    save_jsonl(datas, output_path)
                    print("save sucess!",output_path)
                    datas = []

if __name__ == "__main__":
    # move_code_file()
    # calculate_hash(data_type="text",data_dir="/data0/datasets/MAP-CC/map-cc", is_input_dir=True)
    calculate_hash(num_worker=128)
    # calculate_dup_pairs()
    # calculate_connected_components()
    # calculate_dup_line_each_file()
    # remove_dup()
    # remove_dup_dataset(name="c-sharp")

    # split_map_cc()
