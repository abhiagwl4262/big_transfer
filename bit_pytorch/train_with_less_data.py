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
from sklearn.metrics import confusion_matrix
from datasets import ImageFolder
import torch.nn.functional as F
import glob

try:
    from torch.cuda import amp
    amp_train = True
except:
    amp_train = False

#from .. import bit_common, bit_hyperrule

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

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
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(), 
      #tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      tv.transforms.Normalize((0.43032281,0.49672744 , 0.3134248), (0.08504857, 0.08000449, 0.10248923)),
  ])

  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      #tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      tv.transforms.Normalize((0.43032281,0.49672744 , 0.3134248), (0.08504857, 0.08000449, 0.10248923)),
  ])

  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":

    folder_path = pjoin(args.datadir, "train")
    files  = sorted(glob.glob("%s/*/*.*" % folder_path))
    labels = [int(file.split("/")[-2]) for file in files]
    train_set = ImageFolder(files, labels, train_tx, crop)

    folder_path = pjoin(args.datadir, "val")
    files  = sorted(glob.glob("%s/*/*.*" % folder_path))
    labels = [int(file.split("/")[-2]) for file in files]
    valid_set = ImageFolder(files, labels, val_tx, crop)
    #train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    #valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
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

def select_worst_images(args, model, full_train_loader, device):
  model.eval()
     
  gts   = []
  paths = []
  losses= []

  micro_batch_size = args.batch_size // args.batch_split
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)

  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      #tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      tv.transforms.Normalize((0.43032281,0.49672744 , 0.3134248), (0.08504857, 0.08000449, 0.10248923)),
  ])
  for b, (path, x, y) in enumerate(full_train_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)


      # compute output, measure accuracy and record loss.
      logits = model(x)
    
      paths.extend(path)
      gts.extend(y.cpu().numpy())

      c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)

      losses.extend(c.cpu().numpy().tolist())  # Also ensures a sync point.

    # measure elapsed time
    end = time.time()

  gts    = np.array(gts)
  losses = np.array(losses)
  losses[np.argsort(losses)[int(losses.shape[0]*0.95):]] = 0.0

  paths_ = np.array(paths)[np.where(losses > np.median(losses))[0]]
  gts_   = gts[np.where(losses > np.median(losses))[0]]
  smart_train_set = ImageFolder(paths_, gts_, train_tx, crop)

  smart_train_loader = torch.utils.data.DataLoader(
          smart_train_set, batch_size=micro_batch_size, shuffle=True,
          num_workers=args.workers, pin_memory=True, drop_last=False)

  return smart_train_set, smart_train_loader

def run_eval(model, data_loader, device, chrono, logger, epoch, num_classes):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1, all_top5 = [], [], []
  end = time.time()

  preds = []
  gts   = []

  for b, (path, x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)

        _, preds_ = torch.max(logits, 1)
    
        preds.extend(preds_.cpu().numpy())
        gts.extend(y.cpu().numpy())

        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)

        top1, top5 = topk(logits, y, ks=(1, 5))
        all_c.extend(c.cpu().numpy().tolist())  # Also ensures a sync point.
        all_top1.extend(top1.cpu())
        all_top5.extend(top5.cpu())

    # measure elapsed time
    end = time.time()

  preds = np.array(preds)
  gts   = np.array(gts)

  print("Cij  is equal to the number of observations known to be in group i and predicted to be in group j")
  print(confusion_matrix(gts, preds))

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
  #model = tv.models.resnet50(pretrained=True)
  #model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=classes)
  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  start_epoch = 0

  # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "bit.pth.tar")
  '''try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at epoch {start_epoch}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")'''
  model = model.to(device)
  chrono = lb.Chrono()
  if args.weights:
      model.load_state_dict(torch.load(args.weights)['model'])
  if args.evaluate:
      val_loss, val_acc = run_eval(model, valid_loader, device, chrono, logger, -1, classes)
      return

  optim.zero_grad()
  
  model.train()
  #mixup = bit_hyperrule.get_mixup(len(train_set))
  mixup = -1
  cri = torch.nn.CrossEntropyLoss().to(device)

  logger.info("Starting training!")
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()

  if amp_train:
      cuda = device.type != 'cpu'
      scaler = amp.GradScaler(enabled=cuda)
  epoches = args.epochs 
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=1, epochs=epoches)


  with lb.Uninterrupt() as u:
      for epoch in range(start_epoch, epoches):
          
          model.train()
          
          if epoch == 0:
              pbar = enumerate(train_loader)
              pbar = tqdm.tqdm(pbar, total=len(train_loader))
          else:
              pbar = enumerate(smart_train_loader)
              pbar = tqdm.tqdm(pbar, total=len(smart_train_loader))

          scheduler.step()
          all_top1, all_top5 = [], []
          for param_group in optim.param_groups:
              lr = param_group["lr"]
          #for x, y in recycle(train_loader):
          #for batch_id, (x, y) in enumerate(train_loader):
          for batch_id, (path, x, y) in pbar:


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
                if amp_train:
                    with amp.autocast(enabled=cuda):
                        logits = model(x)
                        if mixup > 0.0:
                            c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
                        else:
                            c = cri(logits, y)
                else:
                    logits = model(x)
                    if mixup > 0.0:
                        c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
                    else:
                        c = cri(logits, y)

            top1, top5 = topk(logits, y, ks=(1, 5))
            all_top1.extend(top1.cpu())
            all_top5.extend(top5.cpu())
            train_loss = c.item()
            train_acc  = np.mean(all_top1)*100.0
            # Accumulate grads
            with chrono.measure("grads"):
              if amp_train:
                  scaler.scale(c / args.batch_split).backward()
              else:
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
                if amp_train:
                    scaler.step(optim)  # optimizer.step
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()
            # Sample new mixup ratio for next batch
            mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

          # Run evaluation and save the model.
          val_loss, val_acc = run_eval(model, valid_loader, device, chrono, logger, epoch, classes)
          _, smart_train_loader = select_worst_images(args, model, train_loader, device)

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
  parser.add_argument("--evaluate", action="store_true")
  parser.add_argument("--weights", type=str, required=False)
  parser.add_argument("--epochs", type=int, required=False)
  main(parser.parse_args())
