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
import torch.nn as nn
import torchvision as tv

import fewshot as fs
import lbtoolbox as lb
import models
import sys
#import bit_pytorch.fewshot as fs
#import bit_pytorch.lbtoolbox as lb
#import bit_pytorch.models as models

import bit_common
import bit_hyperrule
import tqdm

#from .. import bit_common, bit_hyperrule

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


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


def mktrainval(args, logger):
  """Returns train and validation datasets."""
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)

  train_tx = tv.transforms.Compose([
      #tv.transforms.Resize((896, 896)),
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(), 
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  val_tx = tv.transforms.Compose([
      #tv.transforms.Resize((896, 896)),
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")

  micro_batch_size = args.batch_size // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, chrono, logger, epoch):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

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

  model.train()
  logger.info(f"Validation@{epoch} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")
  logger.flush()
  return np.mean(all_c), np.mean(all_top1)*100.0


def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def main(args):

  best_acc = -1

  logger = bit_common.setup_logger(args)
  cp, cn = smooth_BCE(eps=0.1)
  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  classes = 5

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)
  print(len(train_loader))
  logger.info(f"Loading model from {args.model}.npz")
  model = models.KNOWN_MODELS[args.model](head_size=classes, zero_head=True)
  model.load_from(np.load(f"{args.model}.npz"))

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  start_epoch = 0

  # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit.pth.tar")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at epoch {start_epoch}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")

  model = model.to(device)
  optim.zero_grad()

  model.train()
  mixup = bit_hyperrule.get_mixup(len(train_set))
  #mixup = -1
  cri = torch.nn.CrossEntropyLoss().to(device)
  #cri = FocalLoss(cri)
  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()

  epoches = 20
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=1, epochs=epoches)

  with lb.Uninterrupt() as u:
      for epoch in range(start_epoch, epoches):

          pbar = enumerate(train_loader)
          pbar = tqdm.tqdm(pbar, total=len(train_loader))

          scheduler.step()
          all_top1, all_top5 = [], []
          for param_group in optim.param_groups:
              lr = param_group["lr"]
          #for x, y in recycle(train_loader):
          for batch_id, (x, y) in pbar:
          #for batch_id, (x, y) in enumerate(train_loader):
            # measure data loading time, which is spent in the `for` statement.
            chrono._done("load", time.time() - end)

            if u.interrupted:
              break

            # Schedule sending to GPU(s)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Update learning-rate, including stop training if over.
            #lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)            
            #if lr is None:
            #  break

            if mixup > 0.0:
              x, y_a, y_b = mixup_data(x, y, mixup_l)

            # compute output
            with chrono.measure("fprop"):
              logits = model(x)
              top1, top5 = topk(logits, y, ks=(1, 5))
              all_top1.extend(top1.cpu())
              all_top5.extend(top5.cpu())
              if mixup > 0.0:
                c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
              else:
                c = cri(logits, y)
            train_loss = c.item()
            train_acc  = np.mean(all_top1)*100.0
            # Accumulate grads
            with chrono.measure("grads"):
              (c / args.batch_split).backward()
              accum_steps += 1
            accstep = f"({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
            s = f"epoch={epoch} batch {batch_id}{accstep}: loss={train_loss:.5f} train_acc={train_acc:.2f} lr={lr:.1e}"
            #s = f"epoch={epoch} batch {batch_id}{accstep}: loss={c.item():.5f} lr={lr:.1e}"
            pbar.set_description(s)
            #logger.info(f"[batch {batch_id}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
            logger.flush()

            # Update params
            with chrono.measure("update"):
                optim.step()
                optim.zero_grad()
            # Sample new mixup ratio for next batch
            mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

          # Run evaluation and save the model.
          val_loss, val_acc = run_eval(model, valid_loader, device, chrono, logger, epoch)

          best = val_acc > best_acc
          if best:
              best_acc = val_acc
              torch.save({
                  "epoch": epoch,
                  "val_loss": val_loss,
                  "val_acc": val_acc,
                  "train_acc": train_acc,
                  "model": model.state_dict(),
                  "optim" : optim.state_dict(),
              }, savename)
          end = time.time()

  logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  main(parser.parse_args())
