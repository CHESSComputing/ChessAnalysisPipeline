#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : ktrain.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Keras based ML network to train over MNIST dataset
"""

# system modules
import os
import sys
import json
import gzip
import pickle
import argparse

# third-party modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.tools import saved_model_utils


def modelGraph(model_dir):
    """
    Provide input/output names used by TF Graph along with graph itself
    The code is based on TF saved_model_cli.py script.
    """
    input_names = []
    output_names = []
    tag_sets = saved_model_utils.get_saved_model_tag_sets(model_dir)
    for tag_set in sorted(tag_sets):
        print('%r' % ', '.join(sorted(tag_set)))
        meta_graph_def = saved_model_utils.get_meta_graph_def(model_dir, tag_set[0])
        for key in meta_graph_def.signature_def.keys():
            meta = meta_graph_def.signature_def[key]
            if hasattr(meta, 'inputs') and hasattr(meta, 'outputs'):
                inputs = meta.inputs
                outputs = meta.outputs
                input_signatures = list(meta.inputs.values())
                input_names = [signature.name for signature in input_signatures]
                if len(input_names) > 0:
                    output_signatures = list(meta.outputs.values())
                    output_names = [signature.name for signature in output_signatures]
    return input_names, output_names, meta_graph_def

def readData(fin, num_classes):
    """
    Helper function to read MNIST data and provide it to
    upstream code, e.g. to the training layer
    """
    # Load the data and split it between train and test sets
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    f = gzip.open(fin, 'rb')
    if sys.version_info < (3,):
        mnist_data = pickle.load(f)
    else:
        mnist_data = pickle.load(f, encoding='bytes')
    f.close()
    (x_train, y_train), (x_test, y_test) = mnist_data

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def train(fin, fout=None, model_name=None, epochs=1, batch_size=128, h5=False):
    """
    train function for MNIST
    """
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # create ML model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    print("model input", model.input, type(model.input), model.input.__dict__)
    print("model output", model.output, type(model.output), model.output.__dict__)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train model
    x_train, y_train, x_test, y_test = readData(fin, num_classes)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # evaluate trained model
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("save model to", fout)
    writer(fout, model_name, model, input_shape, h5)

def writer(fout, model_name, model, input_shape, h5=False):
    """
    Writer provide write function for given model
    """
    if not fout:
        return
    model.save(fout)
    if h5:
        model.save('{}/{}'.format(fout, h5), save_format='h5')
    pbModel = '{}/saved_model.pb'.format(fout)
    pbtxtModel = '{}/saved_model.pbtxt'.format(fout)
    convert(pbModel, pbtxtModel)

    # get meta-data information about our ML model
    input_names, output_names, model_graph = modelGraph(model_name)
    print("### input", input_names)
    print("### output", output_names)
    # ML uses (28,28,1) shape, i.e. 28x28 black-white images
    # if we'll use color images we'll use shape (28, 28, 3)
    img_channels = input_shape[2]  # last item represent number of colors
    meta = {'name': model_name,
            'model': 'saved_model.pb',
            'labels': 'labels.txt',
            'img_channels': img_channels,
            'input_name': input_names[0].split(':')[0],
            'output_name': output_names[0].split(':')[0],
            'input_node': model.input.name,
            'output_node': model.output.name
    }
    with open(fout+'/params.json', 'w') as ostream:
        ostream.write(json.dumps(meta))
    with open(fout+'/labels.txt', 'w') as ostream:
        for i in range(0, 10):
            ostream.write(str(i)+'\n')
    with open(fout + '/model.graph', 'wb') as ostream:
        ostream.write(model_graph.SerializeToString())

def convert(fin, fout):
    """
    convert input model.pb into output model.pbtxt
    Based on internet search:
    - https://www.tensorflow.org/guide/saved_model
    - https://www.programcreek.com/python/example/123317/tensorflow.core.protobuf.saved_model_pb2.SavedModel
    """
    import google.protobuf
    from tensorflow.core.protobuf import saved_model_pb2
    import tensorflow as tf

    saved_model = saved_model_pb2.SavedModel()

    with open(fin, 'rb') as f:
        saved_model.ParseFromString(f.read())
        
    with open(fout, 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(saved_model))


class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--fin", action="store",
            dest="fin", default="", help="Input MNIST file")
        self.parser.add_argument("--fout", action="store",
            dest="fout", default="", help="Output models area")
        self.parser.add_argument("--model", action="store",
            dest="model", default="mnist", help="model name")
        self.parser.add_argument("--epochs", action="store",
            dest="epochs", default=1, help="number of epochs to use in ML training")
        self.parser.add_argument("--batch_size", action="store",
            dest="batch_size", default=128, help="batch size to use in training")
        self.parser.add_argument("--h5", action="store",
            dest="h5", default="mnist", help="h5 model file name")

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    train(opts.fin, opts.fout,
          model_name=opts.model,
          epochs=opts.epochs,
          batch_size=opts.batch_size,
          h5=opts.h5)

if __name__ == '__main__':
    main()
