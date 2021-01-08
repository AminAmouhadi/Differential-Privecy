# Lint as: python3
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
"""Image SimCLR task definition.

SimCLR training two different modes:
- pretrain
- fine-tuning

For the above two different modes, the following components are different in
the task definition:
- training data format
- training loss
- projection_head and/or supervised_head
"""

import tensorflow as tf

# Lint as: python3
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
"""Image classification task definition."""
from absl import logging
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.beta.dataloaders import classification_input
from official.vision.beta.modeling import factory
from official.vision.beta.projects.simclr.configs import simclr as exp_cfg
from official.vision.beta.tasks import image_classification


@task_factory.register_task_cls(exp_cfg.SimCLRPretrainTask)
class SimCLRPretrainTask(base_task.Task):
  """A task for image classification."""

  def build_model(self):
    pass

  def build_inputs(self, params, input_context=None):
    pass

  def build_losses(self, labels, model_outputs, aux_losses=None):
    pass

  def build_metrics(self, training=True):
    pass

  def train_step(self, inputs, model, optimizer, metrics=None):
    pass

  def validation_step(self, inputs, model, metrics=None):
    pass

  def inference_step(self, inputs, model):
    """Performs the forward step."""
    return model(inputs, training=False)


@task_factory.register_task_cls(exp_cfg.SimCLRFinetuneTask)
class SimCLRFinetuneTask(image_classification.ImageClassificationTask):
  """A task for image classification."""

  def build_model(self):
    pass
