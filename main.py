import os
from os import path
import math
import datetime

import pickle

from Utils.preprocessing import IntentDetectionData, new_preprocessing
from Utils.models import create_model
#
#
import pandas as pd
import numpy as np
#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
#
import bert
# from bert import BertModelLayer
# from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

RANDOM_SEED = 42
#
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)




def prediction(max_length, intents):
    sentences = [
        "Play our song now",
        "how is the weather",
        "hotel booking"
    ]
    bert_model_name = "uncased_L-12_H-768_A-12"
    bert_ckpt_dir = os.path.join("model/", bert_model_name)

    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids + [0] * (max_length - len(tids)), pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))
    models = load_model('model.h5', custom_objects={"BertModelLayer": bert.model.BertModelLayer})

    predictions = models.predict(pred_token_ids).argmax(axis=-1)

    for text, label in zip(sentences, predictions):
        print("text:", text, "\nintent:", intents[label])
    print()


def model_evaluation():
    input_path = path.abspath(os.path.join(__file__, "../", "Data"))
    train = pd.read_csv(input_path + "//train_new_small.csv")
    test = pd.read_csv(input_path + "//test.csv")

    bert_model_name = "uncased_L-12_H-768_A-12"

    bert_ckpt_dir = os.path.join("model/", bert_model_name)
    bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

    tokens = tokenizer.tokenize("I can't wait to visit Bulgaria again!")
    print(tokenizer.convert_tokens_to_ids(tokens))

    classes = train.intent.unique().tolist()
    print(classes)

    data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)
    #######################
    #####################  continue from here ####################
    ############################
    print(data.train_x.shape)

    model = create_model(data.max_seq_len, bert_ckpt_file, bert_config_file, classes)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    print(data.train_x)
    print(data.train_y)
    model.fit(
        x=data.train_x,
        y=data.train_y,
        validation_split=0.1,
        batch_size=16,
        shuffle=True,
        epochs=5
    )

    model.save("Model.h5")


def data_preprocessing():
    input_path = path.abspath(os.path.join(__file__, "../", "Data"))
    train = pd.read_csv(input_path + "//train_new.csv")
    test = pd.read_csv(input_path + "//test.csv")

    intents = train.intent.unique().tolist()
    print(intents)

    return new_preprocessing(train, test, intents)


def persist_model(train_data, test_data, length):
    bert_model_name = "uncased_L-12_H-768_A-12"

    bert_ckpt_dir = os.path.join("model/", bert_model_name)
    bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

    model = create_model(length, bert_ckpt_file, bert_config_file, intents)
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    model.fit(
        x=train_data,
        y=test_data,
        validation_split=0.1,
        batch_size=16,
        shuffle=True,
        epochs=5
    )
    model.save("Model.h5")


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, max_length, intents = data_preprocessing()
    print(x_train)
    #persist_model(x_train, y_train, max_length)
    prediction(max_length, intents)
