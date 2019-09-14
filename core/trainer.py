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
from torchvision.utils import make_grid

from core.utils import set_seed, Progbar, postprocess, set_device
from core.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
  def __init__(self, config, debug=False):
    super().__init__(config, debug=debug)
    # additional things can be set here
    

  # process input and calculate loss every training epoch
  def _train_epoch(self):
    progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
    for images, masks, names in self.train_loader:
      self.model.train()
      # self.adjust_learning_rate()
      self.iteration += 1
      end = time.time()
      inpts = images*(1.-masks)
      inpts, masks = set_device([inpts, masks])
      self.optim.zero_grad()
      pred_img = self.model(inpts, masks)
      complete_img = (pred_img * masks) + (images * (1. - masks))
      
      # calculate loss and update weights for Generator
      loss = self.get_non_gan_loss(pred_img, images, masks)
      loss.backward()
      self.optim.step()
      self.add_summary(self.writer, 'lr/LR', self.optim.param_groups[0]['lr'])

      mae = torch.mean(torch.abs(images - pred_img)) / torch.mean(masks)
      speed = images.size(0)/(time.time() - end)*self.config['world_size']
      logs = [("epoch", self.epoch),("iter", self.iteration),("lr", self.get_lr()),
        ('mae', mae.item()), ('samples/s', speed)]
      if self.config['global_rank'] == 0:
        progbar.add(len(images)*self.config['world_size'], values=logs \
          if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])


  def _eval_epoch(self, it):
    self.model.eval()
    path = os.path.join(self.config['save_dir'], 'samples_{}'.format(str(it).zfill(5)))
    os.makedirs(path, exist_ok=True)
    print('start evaluating ...')
    evaluation_scores = {key: 0 for key,val in self.metrics.items()}
    with torch.no_grad():
      index = 0
      for images, masks, names in self.valid_loader:
        inpts = images*(1.-masks)
        inpts, masks = set_device([inpts, masks])
        output = self.model(inpts, masks)
        orig_imgs = postprocess(images)
        comp_imgs = postprocess(masks*output+(1.-masks)*images)
        pred_imgs = postprocess(output)
        mask_imgs = postprocess(inpts)
        grid_img = make_grid(torch.cat([orig_imgs, mask_imgs, pred_imgs, comp_imgs], dim=0), nrow=4)
        grid_img.save(os.path.join(path, '{}.png'.format(str(index).zfill(5))))
        for key, val in self.metrics.items():
          evaluation_scores[key] += val(orig_imgs, comp_imgs)
        index += 1
      evaluation_message = ' '.join(['{}: {:5f},'.format(key, val/len(self.valid_loader)) \
                          for key,val in evaluation_scores.items()])
      print('[**] Evaluation: {}'.format(evaluation_message))

