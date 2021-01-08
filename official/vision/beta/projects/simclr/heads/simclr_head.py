# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Dense prediction heads."""

# Import libraries
from typing import Text, Optional

import numpy as np
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.projects.simclr.modeling import nn_blocks


@tf.keras.utils.register_keras_serializable(package='simclr')
class ProjectionHead(tf.keras.layers.Layer):
  def __init__(
      self,
      proj_output_dim: int,
      proj_mode: Optional[Text] = 'nonlinear',
      num_proj_layers: int = 3,
      ft_proj_idx: int = 0,
      **kwargs):
    super(ProjectionHead, self).__init__(**kwargs)
    self._proj_output_dim = proj_output_dim
    self._proj_mode = proj_mode
    self._num_proj_layser = num_proj_layers
    self._ft_proj_idx = ft_proj_idx

  def get_config(self):
    config = {
        'proj_output_dim': self._proj_output_dim,
        'proj_mode': self._proj_mode,
        'num_proj_layers': self._num_proj_layser,
        'ft_proj_idx': self._ft_proj_idx
    }
    base_config = super(ProjectionHead, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self._layers = []
    if self._proj_mode is None:
      pass
    elif self._proj_mode == 'linear':
      self._layers.append(
          nn_blocks.DenseBN(
              output_dim=self._proj_output_dim,
              use_bias=False,
              use_normalization=True,
              activation=None,
              name='l_0'
          )
      )
    elif self._proj_mode == 'nonlinear':
      intermediate_dim = int(input_shape[-1])
      for j in range(self._num_proj_layser):
        if j != self._num_proj_layser - 1:
          # for the middle layers, use bias and relu for the output.
          self._layers.append(
              nn_blocks.DenseBN(
                  output_dim=intermediate_dim,
                  use_bias=True,
                  use_normalization=True,
                  activation='relu',
                  name='nl_%d' % j
              )
          )
        else:
          # for the final layer, neither bias nor relu is used.
          self._layers.append(
              nn_blocks.DenseBN(
                  output_dim=self._proj_output_dim,
                  use_bias=False,
                  use_normalization=True,
                  activation=None,
                  name='nl_%d' % j
              )
          )
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          self._proj_mode))

    super(ProjectionHead, self).build(input_shape)

  def call(self, inputs, training=None):
    proj_head_output = None
    proj_finetune_output = None

    if self._proj_mode is None:
      proj_head_output = inputs

    elif self._proj_mode == 'linear':
      assert len(self.linear_layers) == 1, len(self.linear_layers)
      proj_head_output = self._layers[0](inputs, training)
      proj_finetune_output = tf.identity(inputs, 'proj_head_input')

    elif self._proj_mode == 'nonlinear':
      hiddens_list = [tf.identity(inputs, 'proj_head_input')]
      for j in range(self._num_proj_layser):
        hiddens = self._layers[j](hiddens_list[-1], training)
        hiddens_list.append(hiddens)
      proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
      proj_finetune_output = hiddens_list[self._ft_proj_idx]

    return proj_head_output, proj_finetune_output


@tf.keras.utils.register_keras_serializable(package='simclr')
class ClassificationHead(tf.keras.layers.Layer):
  def __init__(
      self,
      num_classes: int,
      name: Text = 'head_supervised',
      **kwargs):
    super(ClassificationHead, self).__init__(name=name, **kwargs)
    self._num_classes = num_classes
    self._name = name

  def get_config(self):
    config = {
        'num_classes': self._num_classes,
    }
    base_config = super(ClassificationHead, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self._dense0 = nn_blocks.DenseBN(self._num_classes)
    super(ClassificationHead, self).build(input_shape)

  def call(self, inputs, training=None):
    inputs = self._dense0(inputs, training)
    inputs = tf.identity(inputs, name='logits_sup')
    return inputs
