import argparse
import json
import multiprocessing
import os
import pickle
from datasketch import MinHash
from nltk import ngrams
import time
from tqdm import tqdm
import gzip
from joblib import Parallel, delayed
import re
import fasttext
from polyglot.text import Text

CONTENT_FIELD_NAME = "text"
DOC_ID_FIELD_NAME = "doc_id"
HASH_FILE_NAME = "hash"
width = 13
bands = 9
r = 13
n_perm = 128

NON_ALPHA = re.compile("[^A-Za-z_0-9]")

fasttext_model = fasttext.load_model("/home/qyhuang/weights/facebook/lid.176.bin")


def get_language_score(text):
    """
    language and score of the language identification model
    """
    global fasttext_model
    labels, scores = fasttext_model.predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    score = min(float(scores[0]), 1.0)
    return {"lang": label, "score": score}


# transform array to byte representation so that it can be hashed
def _H(hs):
    return bytes(hs.byteswap().data)


def generate_hash_values(line, file_name, idx, data_type):
    global width
    global r
    global bands
    ret = []
    try:
        doc_id = f"{file_name}@{idx}"
        json_doc = json.loads(line)
        if CONTENT_FIELD_NAME in json_doc:
            text = json_doc[CONTENT_FIELD_NAME]
        else:
            text = json_doc["content"]
        if data_type == "code":
            text = NON_ALPHA.split(text)
            features = map(lambda x: " ".join(x), ngrams(text, width))
        else:
            # 去除标点符号和特殊字符
            text = re.sub(r'[^\w\s]', '', text)
            text = Text(text).words
            lang = get_language_score(text)["lang"]
            if lang == "zh":
                text = re.sub(r'\s+', '', text).strip()
                features = map(lambda x: "".join(x), ngrams(text, width))
            else:
                text = text.split()
                features = map(lambda x: " ".join(x), ngrams(text, width))

        # features = map(lambda x: "".join(x), ngrams(text, width))
        m = MinHash(num_perm=128)
        [m.update(x.encode("utf8")) for x in features]
        for idx in range(bands):
            save_doc = {}
            save_doc[DOC_ID_FIELD_NAME] = doc_id
            save_doc[HASH_FILE_NAME] = _H(m.hashvalues[idx * r : (idx + 1) * r])
            ret.append(save_doc)
    except Exception as e:
        print(f"procces file {file_name} line {idx} error happe: {e}")
        ret = []
    return ret


def process_file(input_dir, output_dir, num_worker, data_type):
    for file_name in os.listdir(input_dir):
        if file_name == "general_zh_cc.jsonl":
            continue
        file_path = os.path.join(input_dir, file_name)
        file_name = file_path.split("/")[-1]
        # 判断输出文件是否存在
        flag = True
        for band in range(bands):
            output_path = os.path.join(output_dir, str(band), file_name + ".gz")
            # size_g = os.path.getsize(output_path)
            # if size_g<10:
            #     os.remove(output_path)
            if not os.path.exists(output_path):
                flag = False
        if flag:
            print("exixt:", output_path)
            continue

        print(f"process file {file_name}")
        band_hash_value_list = [[] for _ in range(bands)]
        # with gzip.open(file_path, "rt", encoding="utf-8") as f:
        with open(file_path, "r") as f:
            hash_value_lists = Parallel(n_jobs=num_worker)(
                delayed(generate_hash_values)(line, file_name, idx, data_type)
                for idx, line in tqdm(enumerate(f))
            )
            for hash_value_list in hash_value_lists:
                if len(hash_value_list):
                    for idx, hash in enumerate(hash_value_list):
                        band_hash_value_list[idx].append(hash)

            for band in range(bands):
                output_path = os.path.join(output_dir, str(band), file_name + ".gz")
                with open(output_path, "wb") as fout:
                    pickle.dump(band_hash_value_list[band], fout)
            print(f"finish process file {file_name}")


def process_dir(input_dir, output_dir, num_worker, data_type):
    assert data_type in ["text", "code"], "data type error:{data_type}!"
    os.makedirs(output_dir, exist_ok=True)
    for band in range(bands):
        band_path = os.path.join(output_dir, str(band))
        os.makedirs(band_path, exist_ok=True)
    process_file(input_dir, output_dir, num_worker, data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=42, type=int)
    parser.add_argument("--input_dir", default="input_dir")
    parser.add_argument("--output_dir", default="output_dir")
    parser.add_argument("--data_type", default="text", help="text or code")
    parser.add_argument(
        "--content_field_name",
        default=CONTENT_FIELD_NAME,
        type=str,
        help="field name of content in json",
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    process_dir(input_dir, output_dir, args.workers, args.data_type)
