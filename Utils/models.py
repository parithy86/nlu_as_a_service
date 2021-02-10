import tensorflow as tf
import yaml
from tensorflow import keras
import os

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights


def create_model(train_data, test_data, length, intents):
    bert_model_name = "uncased_L-12_H-768_A-12"

    bert_ckpt_dir = os.path.join("model/", bert_model_name)
    bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

    with tf.io.gfile.GFile(bert_config_file, "r") as bert_config:
        bc = StockBertConfig.from_json_string(bert_config.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(length,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(intents), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, length))

    load_stock_weights(bert, bert_ckpt_file)

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

    model_config = [
        {'intents': intents},
        {'max_length': length}]

    with open('model_config.yaml', 'w') as f:
        yaml.dump(model_config, f)
