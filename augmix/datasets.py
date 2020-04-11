# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Original (https://github.com/google-research/augmix/blob/master/imagenet.py)
# modified by Matthew Zurek
#
#

import augmentations

import numpy as np
import torch

args = {
    'aug_prob_coeff': 1.,
    'mixture_width': 3,
    'mixture_depth': -1,
    'aug_severity': 1
}

def aug(image, preprocess, aug_list):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
    aug_list: List of augmentation operations
  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([args['aug_prob_coeff']] * args['mixture_width']))
  m = np.float32(np.random.beta(args['aug_prob_coeff'], args['aug_prob_coeff']))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args['mixture_width']):
    image_aug = image.copy()
    depth = args['mixture_depth'] if args['mixture_depth'] > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args['aug_severity'])
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, aug_list, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.aug_list = aug_list
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess, self.aug_list), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.aug_list),
                  aug(x, self.preprocess, self.aug_list))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)
