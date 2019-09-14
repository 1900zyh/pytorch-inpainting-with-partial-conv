import os
import cv2
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter


from core.dataset import Dataset
from utils.util import set_seed, Progbar, postprocess, set_device
from core import loss as module_loss
from core import metric as module_metric
from core.base_trainer import BaseTrainer
from utils.vgg import Vgg16

class Trainer(BaseTrainer):
  def __init__(self, config, debug=False):
    super().__init__(config, debug=debug)
    # additional things can be set here
    

  # process input and calculate loss every training epoch
  def _train_epoch(self):
    progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
    self.netG.train()
    self.netD.train()
    for name, videos, masks in self.train_loader:
      # self.adjust_learning_rate()
      self.iteration += 1
      end = time.time()
      videos, masks = set_device([videos, masks])
      self.optimG.zero_grad()
      self.optimD.zero_grad()
      pred_img = self.netG(videos, masks)
      complete_img = (pred_img * masks) + (videos * (1. - masks))
      self.add_summary(self.writer_gen, 'lr/LR', self.optimG.param_groups[0]['lr'])
      self.add_summary(self.writer_dis, 'lr/LR', self.optimD.param_groups[0]['lr'])
      
      # calculate loss and update weights for Generator
      gen_loss = self.get_non_gan_loss(pred_img, videos, masks)
      gen_fake = self.netD(complete_img, masks)
      gen_adv_loss = self.adv_loss_fn(gen_fake, True, False)
      gen_loss += gen_adv_loss * self.config['gan_losses']['loss_gan_weight']
      gen_loss.backward()
      self.optimG.step()
      self.add_summary(self.writer_gen, 'loss/adv_loss', gen_adv_loss.item())

      # calculate loss and update weights for Discriminator
      dis_real = self.netD(videos, masks)
      dis_fake = self.netD(complete_img.detach(), masks)
      dis_real_loss = self.adv_loss_fn(dis_real, True, True)
      dis_fake_loss = self.adv_loss_fn(dis_fake, False, True)
      dis_loss = dis_real_loss + dis_fake_loss
      self.add_summary(self.writer_dis, 'loss/dis_fake_loss', dis_fake_loss.item())
      self.add_summary(self.writer_dis, 'loss/dis_real_loss', dis_real_loss.item())
      self.add_summary(self.writer_dis, 'loss/adv_loss', dis_loss.item()/2.)

      dis_loss.backward()
      self.optimD.step()
      mae = torch.mean(torch.abs(videos - pred_img)) / torch.mean(masks)
      speed = videos.size(0)/(time.time() - end)*self.config['world_size']
      logs = [("epoch", self.epoch),("iter", self.iteration),("lr", self.get_lr()),
        ('mae', mae.item()), ('samples/s', speed)]
      if self.config['global_rank'] == 0:
        progbar.add(len(videos)*self.config['world_size'], values=logs \
          if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])


  def _eval_epoch(self, it):
    self.netG.eval()
    path = os.path.join(self.config['save_dir'], 'samples_{}'.format(str(it).zfill(5)))
    os.makedirs(path, exist_ok=True)
    print('start evaluating ...')
    evaluation_scores = {key: 0 for key,val in self.metrics.items()}
    with torch.no_grad():
      index = 0
      for names, videos, masks in self.valid_loader:
        videos, masks = set_device([videos, masks])
        pred_img = self.netG(videos, masks)
        # calculate error
        frames_orig, frames_comp = self.write_to_frames(videos, names, path, pred_img, masks)
        for key, val in self.metrics.items():
          evaluation_scores[key] += val(frames_orig, frames_comp)
      evaluation_message = ' '.join(['{}: {:5f},'.format(key, val/len(self.valid_loader)) \
                          for key,val in evaluation_scores.items()])
      print('[**] Evaluation: {}'.format(evaluation_message))


  def write_to_frames(self, videos, names, path, pred_img, masks):
    default_fps = 6
    complete_img = (pred_img * masks) + (videos * (1. - masks))
    masked_img = videos * (1. - masks) + masks
    # convert videos to frames
    b, t, c, h, w = list(videos.size())
    orig_v = videos.view(-1, c, h, w)
    comp_v = complete_img.view(-1, c, h, w)
    pred_v = pred_img.view(-1, c, h, w)
    mask_v = masked_img.view(-1, c, h, w)
    # save all frames
    frames_mask = postprocess(mask_v)
    frames_pred = postprocess(pred_v)
    for i in range(b):
      video_name = os.path.dirname(names[0][i])
      os.makedirs(os.path.join(path, video_name), exist_ok=True)
      mask_writer = cv2.VideoWriter(os.path.join(path, video_name, 'mask.avi'),
        cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
      pred_writer = cv2.VideoWriter(os.path.join(path, video_name, 'pred.avi'),
        cv2.VideoWriter_fourcc(*"MJPG"), default_fps, (w, h))
      for j in range(t):
        img_pred = Image.fromarray(frames_pred[i*t+j])
        pred_writer.write(cv2.cvtColor(np.array(img_pred), cv2.COLOR_RGB2BGR))
        img_mask = Image.fromarray(frames_mask[i*t+j])
        mask_writer.write(cv2.cvtColor(np.array(img_mask), cv2.COLOR_RGB2BGR))
      pred_writer.release()
      mask_writer.release()
    # return for evaluation 
    frames_orig = postprocess(orig_v)
    frames_comp = postprocess(comp_v)
    return frames_orig, frames_comp
