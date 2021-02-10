import os
import bert
import numpy as np
import yaml
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from bert.tokenization.bert_tokenization import FullTokenizer

from Utils.preprocessing import data_preprocessing
from Utils.models import create_model


# RANDOM_SEED = 42
# #
# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)


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
    print(predictions)

    for text, label in zip(sentences, predictions):
        print("text:", text, "\nintent:", intents[label])
    print()


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
    # x_train, x_test, y_train, y_test, max_length, intents = data_preprocessing()
    # print(x_train)
    # persist_model(x_train, y_train, max_length)
    max_length = 29
    intents = ['PlayMusic', 'AddToPlaylist', 'RateBook', 'SearchScreeningEvent', 'BookRestaurant', 'GetWeather',
               'SearchCreativeWork']

    model_config = [
        {'intents': intents},
        {'max_length': max_length}]

    with open('model_config.yaml', 'w') as f:
        yaml.dump(model_config, f)

    # with open(r'E:\data\store_file.yaml', 'w') as file:
    #     documents = yaml.dump(model_config, file)

    # print(max_length, intents)
    prediction(max_length, intents)
