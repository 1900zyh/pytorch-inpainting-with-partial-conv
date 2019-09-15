import os
import cv2
import time
import math
import glob
import shutil
import datetime
import numpy as np
from PIL import Image
from math import log10

import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.utils import make_grid, save_image

from core.utils import set_seed, Progbar, set_device
from core.utils import postprocess, unnormalize
from core.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
  def __init__(self, config, debug=False):
    super().__init__(config, debug=debug)
    # additional things can be set here
    self.pretrain = True
    if debug:
      self.config['trainer']['save_freq'] = 10
      self.config['lr_scheduler']['step_size'] = 0

  # process input and calculate loss every training epoch
  def _train_epoch(self):
    progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
    for images, masks, names in self.train_loader:
      self.adjust_learning_rate()
      # self.adjust_learning_rate()
      self.iteration += 1
      end = time.time()
      inpts = images*masks
      images, inpts, masks = set_device([images, inpts, masks])
      self.optim.zero_grad()
      pred_img, _ = self.model(inpts, masks)
      complete_img = pred_img * (1.-masks) + images * masks
      
      # calculate loss and update weights for Generator
      loss = self.get_non_gan_loss(pred_img, images, masks)
      loss.backward()
      self.optim.step()
      self.add_summary(self.writer, 'lr/LR', self.optim.param_groups[0]['lr'])

      # logs
      mae = torch.mean(torch.abs(images - pred_img)) / torch.mean(masks)
      speed = images.size(0)/(time.time() - end)*self.config['world_size']
      logs = [("epoch", self.epoch),("iter", self.iteration),("lr", self.get_lr()),
        ('mae', mae.item()), ('samples/s', speed)]
      if self.config['global_rank'] == 0:
        progbar.add(len(images)*self.config['world_size'], values=logs \
          if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])

      # saving and evaluating
      if self.iteration % self.train_args['save_freq'] == 0:
        self._save(self.iteration//self.train_args['save_freq'])
        self._eval_epoch(self.iteration//self.train_args['save_freq'])
        if self.config['global_rank'] == 0:
          print('[**] Training till {} in Rank {}\n'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.config['global_rank']))
      if self.iteration > self.config['trainer']['iterations']:
        break

  def _eval_epoch(self, it):
    self.model.eval()
    self.valid_sampler.set_epoch(it)
    path = os.path.join(self.config['save_dir'], 'samples_{}'.format(str(it).zfill(5)))
    os.makedirs(path, exist_ok=True)
    if self.config['global_rank'] == 0:
      print('start evaluating ...')
    evaluation_scores = {key: 0 for key,val in self.metrics.items()}
    index = 0
    for images, masks, names in self.valid_loader:
      inpts = images*masks
      images, inpts, masks = set_device([images, inpts, masks])
      with torch.no_grad():
        output, _ = self.model(inpts, masks)
      grid_img = make_grid(torch.cat([unnormalize(images), unnormalize(masks*images),
        unnormalize(output), unnormalize(masks*images+(1-masks)*output)], dim=0), nrow=4)
      save_image(grid_img, os.path.join(path, '{}'.format(names[0])))
      orig_imgs = postprocess(images)
      comp_imgs = postprocess(masks*images+(1-masks)*output)
      for key, val in self.metrics.items():
        evaluation_scores[key] += val(orig_imgs, comp_imgs)
      index += 1
    for key, val in evaluation_scores.items():
      tensor = set_device(torch.FloatTensor([val/index]))
      dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
      evaluation_scores[key] = tensor.cpu().item()
    evaluation_message = ' '.join(['{}: {:5f},'.format(key, val/self.config['world_size']) \
                        for key,val in evaluation_scores.items()])
    if self.config['global_rank'] == 0:
      print('[**] Evaluation: {}'.format(evaluation_message))


  def adjust_learning_rate(self,):
    if self.pretrain and self.iteration > self.config['lr_scheduler']['step_size']:
      self.pretrain = False
      for param_group in self.optim.param_groups:
        param_group['lr'] = 5e-5
      for name, module in self.model.named_modules():
        if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm)) and 'enc' in name:
          module.eval()
      
      
