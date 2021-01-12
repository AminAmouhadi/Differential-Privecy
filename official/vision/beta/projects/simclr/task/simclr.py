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
from typing import Dict
from absl import logging
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.beta.modeling import factory
from official.vision.beta.modeling import backbones
from official.vision.beta.projects.simclr.configs import simclr as exp_cfg
from official.vision.beta.tasks import image_classification
from official.vision.beta.projects.simclr.modeling import simclr_model
from official.vision.beta.projects.simclr.heads import simclr_head
from official.vision.beta.projects.simclr.dataloaders import simclr_input
from official.vision.beta.projects.simclr.losses import contractive_losses


# TODO: Weight decay and regularization ?
@task_factory.register_task_cls(exp_cfg.SimCLRPretrainTask)
class SimCLRPretrainTask(base_task.Task):
  """A task for image classification."""

  @staticmethod
  def _update_train_metrics(
      train_metrics: Dict[str, tf.keras.metrics.Metric],
      train_losses: Dict[str, tf.Tensor]):
    """Updated training metrics."""
    total_loss_m = train_metrics['total_loss']
    contrast_loss_m = train_metrics['contrast_loss']
    contrast_acc_m = train_metrics['contrast_acc']
    contrast_entropy_m = train_metrics['contrast_entropy']

    total_loss = train_losses['total_loss']
    contrast_loss = train_losses['contrast_loss']
    contrast_logits = train_losses['contrast_logits']
    contrast_labels = train_losses['contrast_labels']

    contrast_loss_m.update_state(contrast_loss)
    total_loss_m.update_state(total_loss)

    contrast_acc_val = tf.equal(
        tf.argmax(contrast_labels, 1), tf.argmax(contrast_logits, axis=1))
    contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
    contrast_acc_m.update_state(contrast_acc_val)

    prob_con = tf.nn.softmax(contrast_logits)
    entropy_con = -tf.reduce_mean(
        tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
    contrast_entropy_m.update_state(entropy_con)

  def build_model(self):
    model_config = self.task_config.model
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + model_config.input_size)

    l2_weight_decay = self.task_config.weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        model_config=model_config.backbone,
        l2_regularizer=l2_regularizer
    )

    projection_head_config = model_config.projection_head
    projection_head = simclr_head.ProjectionHead(
        proj_output_dim=projection_head_config.proj_output_dim,
        proj_mode=projection_head_config.proj_mode,
        num_proj_layers=projection_head_config.num_proj_layers,
        ft_proj_idx=projection_head_config.ft_proj_idx)

    supervised_head_config = model_config.projection_head
    supervised_head = simclr_head.ClassificationHead(
        num_classes=supervised_head_config.num_classes)

    model = simclr_model.SimCLRModel(
        input_specs=input_specs,
        backbone=backbone,
        projection_head=projection_head,
        supervised_head=supervised_head,
        mode='pretrain')

    return model

  def build_inputs(self, params, input_context=None):
    input_size = self.task_config.model.input_size

    decoder = simclr_input.Decoder(params.decoder.decode_label)
    parser = simclr_input.Parser(
        output_size=input_size[:2],
        aug_rand_crop=params.parser.aug_rang_crop,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_color_distort=params.parser.aug_color_distort,
        aug_color_jitter_strength=params.parser.aug_color_jitter_strength,
        aug_color_jitter_impl=params.parser.aug_color_jitter_impl,
        aug_rand_blur=params.parser.aug_rand_blur,
        parse_label=params.parser.parse_label,
        test_crop=params.parser.test_crop,
        mode=params.parser.mode,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(
      self,
      labels,
      projection_outputs,
      aux_losses=None) -> Dict[str, tf.Tensor]:
    losses_config = self.task_config.loss
    losses_obj = contractive_losses.ContrastiveLoss(
        projection_norm=losses_config.projection_norm,
        temperature=losses_config.temperature)
    # The projection outputs from model has the size of
    # (2 * bsz, project_dim)
    projection1, projection2 = tf.split(projection_outputs, 2, 0)
    con_loss, logits_con, labels_con = losses_obj(
        projection1=projection1,
        projection2=projection2)

    total_loss = tf_utils.safe_mean(con_loss)
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    losses = {
        'total_loss': total_loss,
        'contrast_loss': con_loss,
        'contrast_logits': logits_con,
        'contrast_labels': labels_con
    }
    return losses

  def build_metrics(self, training=True) -> Dict[str, tf.keras.metrics.Metric]:

    if training:
      metrics = {}
      metric_names = [
          'total_loss',
          'contrast_loss',
          'contrast_acc',
          'contrast_entropy'
      ]
      for name in metric_names:
        metrics[name] = tf.keras.metrics.Mean(name, dtype=tf.float32)
    else:
      k = self.task_config.evaluation.top_k
      if self.task_config.evaluation.one_hot:
        metrics = {
            'top_1_accuracy': tf.keras.metrics.CategoricalAccuracy(
                name='top_1_accuracy'),
            'top_k_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))
        }
      else:
        metrics = {
            'top_1_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(
                name='top_1_accuracy'),
            'top_k_accuracy': tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))
        }
    return metrics

  def train_step(self, inputs, model, optimizer, metrics=None):
    features, labels = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      for item in outputs:
        outputs[item] = tf.nest.map_structure(
            lambda x: tf.cast(x, tf.float32), outputs[item])

      # Computes per-replica loss.
      losses = self.build_losses(
          projection_outputs=outputs['projection_outputs'],
          labels=labels, aux_losses=model.losses)

      scaled_loss = losses['total_loss'] / num_replicas
      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient when LossScaleOptimizer is used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: losses['total_loss']}

    self._update_train_metrics(train_metrics=metrics, train_losses=losses)

    for m in metrics:
      logs.update({m: metrics[m].result()})

    return logs

  def validation_step(self, inputs, model, metrics=None):
    features, labels = inputs
    if self.task_config.evaludation.one_hot:
      num_classes = self.task_config.model.supervised_head.num_classes
      labels = tf.one_hot(labels, num_classes)

    outputs = self.inference_step(features, model)
    for item in outputs:
      outputs[item] = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs[item])

    losses = self.build_losses(
        projection_outputs=outputs['projection_outputs'],
        labels=labels, aux_losses=model.losses)

    logs = {self.loss: losses['total_loss']}

    for m in metrics:
      metrics[m].update_state(labels, outputs['supervised_outputs'])

    for m in metrics:
      logs.update({m: metrics[m].result()})

    return logs

  def inference_step(self, inputs, model):
    """Performs the forward step."""
    return model(inputs, training=False)


@task_factory.register_task_cls(exp_cfg.SimCLRFinetuneTask)
class SimCLRFinetuneTask(image_classification.ImageClassificationTask):
  """A task for image classification."""

  def build_model(self):
    pass
