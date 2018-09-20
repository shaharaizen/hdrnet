# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Useful image metrics."""

import tensorflow as tf

import numpy as np

def total_loss(target, prediction, name=None):
  """
  calculates the total loss: a combination of the l2 loss and l2 loss of the magnitudes
  """
  return tf.add(l2_loss(target, prediction, name), 0.1 * gradient_loss(target, prediction, name))

def l2_loss(target, prediction, name=None):
  with tf.name_scope(name, default_name='l2_loss', values=[target, prediction]):
    loss = tf.reduce_mean(tf.square(target-prediction))
  return loss

def gradient_loss(target, prediction,name=None):
  """
  calculates l2 loss between the magnitudes
  """
  with tf.name_scope(name, default_name='gradient_loss', values=[target, prediction]):
    loss = l2_loss(calc_gradient(target), calc_gradient(prediction))
  return loss

def calc_gradient(img):
  """
  calculates the magnitude of the image
  """
  img_left = img[:,:-1,:]
  img_right = img[:, 1:, :]
  img_up = img[:,:, :-1]
  img_down = img[:,:, 1:]
  x_grad = (img_left - img_right)[:,:,1:]
  y_grad = (img_up - img_down)[:,1:,:]

  magnitude = tf.sqrt(tf.square(x_grad) + tf.square(y_grad) + 0.00000000001)

  return magnitude

def psnr(target, prediction, name=None):
  with tf.name_scope(name, default_name='psnr_op', values=[target, prediction]):
    squares = tf.square(target-prediction, name='squares')
    squares = tf.reshape(squares, [tf.shape(squares)[0], -1])
    # mean psnr over a batch
    p = tf.reduce_mean((-10/np.log(10))*tf.log(tf.reduce_mean(squares, axis=[1])))
  return p
