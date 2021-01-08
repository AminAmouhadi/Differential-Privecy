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
"""Data parser and processing for SimCLR.

For pre-training:
- Preprocessing:
  -> random cropping
  -> resize back to the original size
  -> random color distortions
  -> random Gaussian blur (sequential)
- Each image need to be processed randomly twice

```snippets
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
```

For fine-tuning:
typical image classification input
"""

from typing import List, Optional
# Import libraries
import tensorflow as tf

from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import augment
from official.vision.beta.ops import preprocess_ops


class Parser(parser.Parser):
  """Parser for SimCLR training."""

  def __init__(self,
               output_size: List[int],
               num_classes: float,
               dtype: str = 'float32'):
    pass

  def _parse_train_data(self, decoded_tensors):
    pass

  def _parse_eval_data(self, decoded_tensors):
    """Parses data for evaluation."""
    pass
