#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : mnist_img.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: 
"""

import json
import gzip
import argparse
# from itertools import chain

import numpy as np
import matplotlib.pyplot as plt


def readImage(fname, fout, num_images=5, imgId=2):
    """
    Helper function to read MNIST image
    """
    image_size = 28
    with gzip.open(fname, 'r') as fstream:
        fstream.read(16)
        buf = fstream.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
        image = np.asarray(data[imgId]).squeeze()
        plt.imsave(fout, image)
        print("read:", fname, "wrote:", fout, "image:", type(image), "shape:", image.shape)

def img2json(image):
    """
    Convert given image to JSON data format used by TFaaS
    """
    # values = [int(i) for i in list(chain.from_iterable(image))]
    # values = image.tolist()
    values = []
    for row in image.tolist():
        row = [int(i) for i in row]
        vals = [[i] for i in row]
        values.append(vals)
    # final values should be an array of elements, e.g. single image representation
    values = [values]
    keys = [str(i) for i in range(0, 10)]
    meta = {
        'keys': keys,
        'values': values,
        'model': 'mnist'
    }
    with open('img.json', 'w') as ostream:
        ostream.write(json.dumps(meta))


class OptionParser():
    def __init__(self):
        "User based option parser"
        fname = "train-images-idx3-ubyte.gz"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--fin", action="store",
            dest="fin", default=fname, help=f"Input MNIST file, default {fname}")
        self.parser.add_argument("--fout", action="store",
            dest="fout", default="img.png", help="Output image fila name, default img.png")
        self.parser.add_argument("--nimages", action="store",
            dest="nimages", default=5, help="number of images to read, default 5")
        self.parser.add_argument("--imgid", action="store",
            dest="imgid", default=2, help="image index to use from nimages, default 2 (number 4)")

def main():
    """
    main function to produce image file from mnist dataset.
    MNIST dataset can be downloaded from
    curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    """
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    num_images = int(opts.nimages)
    imgId = int(opts.imgid)
    img = readImage(opts.fin, opts.fout, num_images, imgId)

if __name__ == '__main__':
    main()
