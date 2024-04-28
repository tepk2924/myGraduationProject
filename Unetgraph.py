#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import tensorflow as tf

def build_graph():

    BC = 64
    #아래에 나오는 주석들은 BC가 64 채널일 때의 사이즈임.
    input_tensor = tf.keras.Input((480, 640, 4)) #(480, 640, 4)
    relu = tf.keras.layers.ReLU()
    layer00 = tf.expand_dims(tf.pad(input_tensor, ((94, 94), (94, 94), (0, 0)), mode="SYMMETRIC"), axis=0)
    layer01 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(3, 3))(layer00))
    layer02 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(3, 3))(layer01))
    layer10 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer02)
    layer11 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer10))
    layer12 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer11))
    layer20 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer12)
    layer21 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer20))
    layer22 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer21))
    layer30 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer22)
    layer31 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer30))
    layer32 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer31))
    layer40 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer32)
    layer41 = relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer40))
    layer42 = relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer41))
    layer32_cropped = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(layer32)
    layer42_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer42)
    layer42_conved22 = tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(2, 2), padding="same")(layer42_upsample)
    layer50 = tf.keras.layers.Concatenate(axis=1)((layer32_cropped, layer42_conved22))
    layer51 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer50))
    layer52 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer51))
    layer22_cropped = tf.keras.layers.Cropping2D(cropping=((16, 16), (16, 16)))(layer22)
    layer52_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer52)
    layer52_conved22 = tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(2, 2), padding="same")(layer52_upsample)
    layer60 = tf.keras.layers.Concatenate(axis=1)((layer22_cropped, layer52_conved22))
    layer61 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer60))
    layer62 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer61))
    layer12_cropped = tf.keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(layer12)
    layer62_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer62)
    layer62_conved22 = tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(2, 2), padding="same")(layer62_upsample)
    layer70 = tf.keras.layers.Concatenate(axis=1)((layer12_cropped, layer62_conved22))
    layer71 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer70))
    layer72 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer71))
    layer02_cropped = tf.keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(layer02)
    layer72_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer72)
    layer72_conved22 = tf.keras.layers.Conv2D(filters=BC, kernel_size=(2, 2), padding="same")(layer72_upsample)
    layer80 = tf.keras.layers.Concatenate(axis=1)((layer02_cropped, layer72_conved22))
    layer81 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(5, 5))(layer80))
    layer82 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(5, 5))(layer81))
    output_tensor = tf.squeeze(tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1))(layer82))

    return input_tensor, output_tensor