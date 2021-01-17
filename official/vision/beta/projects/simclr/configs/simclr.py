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
import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common
from official.vision.beta.projects.simclr.modeling import simclr_model


@dataclasses.dataclass
class Decoder(hyperparams.Config):
  decode_label: bool = True


@dataclasses.dataclass
class Parser(hyperparams.Config):
  aug_rand_crop: bool = True,
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
  supervised_head = SupervisedHead(num_classes=1001)
  norm_activation: common.NormActivation = common.NormActivation(
      use_sync_bn=False)
  mode: str = simclr_model.PRETRAIN


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
