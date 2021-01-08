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
"""Build simclr models."""

# Import libraries
import tensorflow as tf

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package="simclr")
class SimCLRModel(tf.keras.Model):
  """A classification model based on SimCLR framework."""

  def __init__(self,
               backbone,
               projection_head,
               supervised_head,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               mode: str = 'pretrain',
               **kwargs):
    super(SimCLRModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'projection_head': projection_head,
        'supervised_head': supervised_head,
        'input_specs': input_specs,
        'mode': mode,
    }
    self._input_specs = input_specs
    self._back_bone = backbone
    self._projection_head = projection_head
    self._supervised_head = supervised_head
    self._mode = mode

  def call(self, inputs, training=None, **kwargs):
    model_outputs = {}

    if training and self._mode == 'pretrain':
      num_transforms = 2
    else:
      num_transforms = 1

    # Split channels, and optionally apply extra batched augmentation.
    # (bsz, h, w, c*num_transforms) -> [(bsz, h, w, c), ....]
    features_list = tf.split(inputs, num_or_size_splits=num_transforms, axis=-1)
    # (num_transforms * bsz, h, w, c)
    features = tf.concat(features_list, 0)

    # Base network forward pass.
    projections = self._back_bone(features, training=training)

    # Add heads.
    projection_outputs, supervised_inputs = self._projection_head(
        projections, training)

    if self._supervised_head:
      supervised_outputs = self._supervised_head(supervised_inputs)
    else:
      supervised_outputs = None

    model_outputs.update({
        'projection_outputs': projection_outputs,
        'supervised_outputs': supervised_outputs
    })

    return projection_outputs, supervised_outputs

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
