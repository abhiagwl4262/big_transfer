# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv

import fewshot as fs
import lbtoolbox as lb
import models

#import bit_pytorch.fewshot as fs
#import bit_pytorch.lbtoolbox as lb
#import bit_pytorch.models as models

import bit_common
import bit_hyperrule
from PIL import Image
import os

#from .. import bit_common, bit_hyperrule

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i


def mkval(args):
  """Returns train and validation datasets."""
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)


  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if args.dataset == "cifar10":
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  micro_batch_size = args.batch_size // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  return valid_set, valid_loader


def run_eval(model, data_loader, device, chrono,  step):
  # switch to evaluate mode
  model.eval()

  all_c, all_top1, all_top5 = [], [], []
  end = time.time()
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)
        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
        top1, top5 = topk(logits, y, ks=(1, 5))
        all_c.extend(c.cpu())  # Also ensures a sync point.
        all_top1.extend(top1.cpu())
        all_top5.extend(top5.cpu())

    # measure elapsed time
    end = time.time()

  print(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")
  return all_c, all_top1, all_top5


def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

def run_predict(model, test_path, data_transform, device):
    f_write = open("submission.csv", "a")
    f_write.write('image_id,label\n')
    img_paths = [img for img in os.listdir(test_path)]
    for path in img_paths:
        img_path = os.path.join(test_path,path)
        img      = Image.open(img_path)
        img      = data_transform(img)

        pred     = model(img.unsqueeze(0))
        class_id = torch.argmax(pred).item()
        f_write.write(path + "," + str(class_id) + "\n")

    f_write.close()
def main(args):

  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #classes = 5

  #valid_set, valid_loader = mkval(args)

  model = models.KNOWN_MODELS[args.model](head_size=args.classes, zero_head=False)
  model = torch.nn.DataParallel(model)
  checkpoint = torch.load(args.weight_path, map_location=device)
  model.load_state_dict(checkpoint["model"])

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.

  # Note: no weight-decay!

  model = model.to(device)

  model.eval()

  chrono = lb.Chrono()
  #run_eval(model, valid_loader, device, chrono, step='end')
  end = time.time()
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((448, 448)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  run_predict(model, args.datadir, val_tx, device) 

if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--weight_path", required=True,
                      help="Path to the trained model")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--classes", type=int, default=5, 
                      help="Number of classes in the data")
  main(parser.parse_args())
