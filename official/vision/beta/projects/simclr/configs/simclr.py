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
"""SimCLR configurations"""
from typing import List, Optional
import os
import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common
from official.vision.beta.projects.simclr.modeling import simclr_model


@dataclasses.dataclass
class Decoder(hyperparams.Config):
  decode_label: bool = True


@dataclasses.dataclass
class Parser(hyperparams.Config):
  aug_rand_crop: bool = True
  aug_rand_hflip: bool = True
  aug_color_distort: bool = True
  aug_color_jitter_strength: float = 1.0
  aug_color_jitter_impl: str = 'simclrv2'
  aug_rand_blur: bool = True
  parse_label: bool = True
  test_crop: bool = True
  mode: str = simclr_model.PRETRAIN


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  # simclr specific configs
  parser: Parser = Parser()
  decoder: Decoder = Decoder()


@dataclasses.dataclass
class ProjectionHead(hyperparams.Config):
  proj_output_dim: int = 128
  num_proj_layers: int = 3
  ft_proj_idx: int = 0


@dataclasses.dataclass
class SupervisedHead(hyperparams.Config):
  num_classes: int = 1001


@dataclasses.dataclass
class ContrastiveLoss(hyperparams.Config):
  projection_norm: bool = True
  temperature: float = 1.0


@dataclasses.dataclass
class ClassificationLosses(hyperparams.Config):
  label_smoothing: float = 0.0
  one_hot: bool = True


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  top_k: int = 5
  one_hot: bool = True


@dataclasses.dataclass
class SimCLRModel(hyperparams.Config):
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  projection_head: ProjectionHead = ProjectionHead(
      proj_output_dim=128,
      num_proj_layers=3,
      ft_proj_idx=0)
  supervised_head: SupervisedHead = SupervisedHead(num_classes=1001)
  norm_activation: common.NormActivation = common.NormActivation(
      use_sync_bn=False)
  mode: str = simclr_model.PRETRAIN
  backbone_trainable: bool = True


@dataclasses.dataclass
class SimCLRPretrainTask(cfg.TaskConfig):
  model: SimCLRModel = SimCLRModel(mode=simclr_model.PRETRAIN)
  train_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.PRETRAIN), is_training=True)
  validation_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.PRETRAIN), is_training=False)
  loss: ContrastiveLoss = ContrastiveLoss()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  # all or backbone
  init_checkpoint_modules: str = 'all'
  optimizer: str = 'lars'
  weight_decay: float = 1e-6


@dataclasses.dataclass
class SimCLRFinetuneTask(cfg.TaskConfig):
  model: SimCLRModel = SimCLRModel(mode=simclr_model.FINETUNE)
  train_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.FINETUNE), is_training=True)
  validation_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.FINETUNE), is_training=False)
  loss: ClassificationLosses = ClassificationLosses()
  init_checkpoint: Optional[str] = None
  # all, backbone_projection or backbone
  init_checkpoint_modules: str = 'backbone_projection'
  optimizer: str = 'sgd'
  weight_decay: float = 1e-6


@exp_factory.register_config_factory('simclr_pretraining')
def simclr_pretraining() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=SimCLRPretrainTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


@exp_factory.register_config_factory('simclr_finetuning')
def simclr_finetuning() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=SimCLRFinetuneTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


@exp_factory.register_config_factory('simclr_pretraining_imagenet')
def simclr_pretraining_imagenet() -> cfg.ExperimentConfig:
  """Image classification general."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  return cfg.ExperimentConfig(
      task=SimCLRPretrainTask(
          model=SimCLRModel(
              mode=simclr_model.PRETRAIN,
              backbone_trainable=True,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)),
              projection_head=ProjectionHead(
                  proj_output_dim=128,
                  num_proj_layers=3,
                  ft_proj_idx=1),
              supervised_head=SupervisedHead(num_classes=1001),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
          loss=ContrastiveLoss(),
          evaluation=Evaluation(),
          train_data=DataConfig(
              parser=Parser(mode=simclr_model.PRETRAIN),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              parser=Parser(mode=simclr_model.PRETRAIN),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size),
          weight_decay=1e-6,
          optimizer='lars'
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=90 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'lars',
                  'lars': {
                      'momentum': 0.9,
                      'weight_decay_rate': 0.0,
                      'exclude_from_weight_decay': [
                          'batch_normalization', 'bias', 'head_supervised']
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          30 * steps_per_epoch, 60 * steps_per_epoch,
                          80 * steps_per_epoch
                      ],
                      'values': [
                          0.1 * train_batch_size / 256,
                          0.01 * train_batch_size / 256,
                          0.001 * train_batch_size / 256,
                          0.0001 * train_batch_size / 256,
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


@exp_factory.register_config_factory('simclr_finetuning_imagenet')
def simclr_finetuning_imagenet() -> cfg.ExperimentConfig:
  """Image classification general."""
  """Image classification general."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  pretrain_model_base = ""
  return cfg.ExperimentConfig(
      task=SimCLRFinetuneTask(
          model=SimCLRModel(
              mode=simclr_model.FINETUNE,
              backbone_trainable=True,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)),
              projection_head=ProjectionHead(
                  proj_output_dim=128,
                  num_proj_layers=3,
                  ft_proj_idx=1),
              supervised_head=SupervisedHead(num_classes=1001),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
          loss=ClassificationLosses(),
          train_data=DataConfig(
              parser=Parser(mode=simclr_model.FINETUNE),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              parser=Parser(mode=simclr_model.FINETUNE),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size),
          weight_decay=1e-6,
          optimizer='lars',
          init_checkpoint=pretrain_model_base,
          # all, backbone_projection or backbone
          init_checkpoint_modules='backbone_projection'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=90 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'lars',
                  'lars': {
                      'momentum': 0.9,
                      'weight_decay_rate': 0.0,
                      'exclude_from_weight_decay': [
                          'batch_normalization', 'bias', 'head_supervised']
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          30 * steps_per_epoch, 60 * steps_per_epoch,
                          80 * steps_per_epoch
                      ],
                      'values': [
                          0.1 * train_batch_size / 256,
                          0.01 * train_batch_size / 256,
                          0.001 * train_batch_size / 256,
                          0.0001 * train_batch_size / 256,
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
