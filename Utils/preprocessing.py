import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
from tqdm import tqdm
import pandas as pd
import os
from os import path

DATA_COLUMN = "text"
LABEL_COLUMN = "intent"


def data_preprocessing():

    input_path = path.abspath(os.path.join(__file__, "../../", "Data"))
    train = pd.read_csv(input_path + "//train_new.csv")
    test = pd.read_csv(input_path + "//test.csv")

    intents = train.intent.unique().tolist()
    print(intents)

    bert_model_name = "uncased_L-12_H-768_A-12"
    bert_ckpt_dir = os.path.join("model/", bert_model_name)

    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
    train_new, test_new = map(lambda df: df.reindex(df[DATA_COLUMN].str.len().sort_values().index),
                              [train, test])

    train_x, train_y, max_length_train = prepare_data(train_new, intents, tokenizer)
    test_x, test_y, max_length_test = prepare_data(test_new, intents, tokenizer)

    max_length_1 = min(max_length_train, max_length_test)

    train_x_new = add_padding(train_x, max_length_1)
    test_x_new = add_padding(test_x, max_length_1)
    return train_x_new, test_x_new, train_y, test_y, max_length_1, intents


def prepare_data(data, intents, tokenizer):
    max_seq_len = 0
    x, y = [], []

    for _, row in tqdm(data.iterrows()):
        text, label = row[DATA_COLUMN], row[LABEL_COLUMN]
        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        max_seq_len = max(max_seq_len, len(token_ids))
        x.append(token_ids)
        y.append(intents.index(label))
    print(max_seq_len)
    return np.array(x), np.array(y), max_seq_len


def add_padding(ids, max_length):
    x = []
    for input_ids in ids:
        input_ids = input_ids[:min(len(input_ids), max_length - 2)]
        input_ids = input_ids + [0] * (max_length - len(input_ids))
        x.append(np.array(input_ids))
    return np.array(x)
