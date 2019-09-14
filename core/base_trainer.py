import os
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
from core.model import PConvUNet
from utils.util import set_seed, set_device
from utils.vgg import Vgg16

from core import loss as module_loss
from core import metric as module_metric



class BaseTrainer():
  def __init__(self, config, debug=False):
    self.config = config
    self.epoch = 0
    self.iteration = 0
    self.vgg = Vgg16(requires_grad=False)
    self.vgg = set_device(self.vgg)

    # setup data set and data loader
    self.train_dataset = Dataset(config['data_loader'], debug=debug, split='train')
    self.valid_dataset = Dataset(config['data_loader'], debug=debug, split='valid')
    worker_init_fn = partial(set_seed, base=config['seed'])
    self.train_sampler = None
    if config['distributed']:
      self.train_sampler = DistributedSampler(self.train_dataset, 
        num_replicas=config['world_size'], rank=config['global_rank'])
    self.train_loader = DataLoader(self.train_dataset, 
      batch_size= config['data_loader']['batch_size'] // config['world_size'],
      shuffle=(self.train_sampler is None), num_workers=4,
      pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)
    self.valid_loader = DataLoader(dataset=self.valid_dataset, 
      batch_size=config['data_loader']['batch_size'], shuffle=False)

    # set loss functions and evaluation metrics
    self.losses = {entry['nickname']: (
        getattr(module_loss, entry['type'])(**entry['args']),
        entry['weight'], 
        entry['input']
      ) for entry in config['losses']}
    self.metrics = {met: getattr(module_metric, met) for met in config['metrics']}

    # setup models including generator and discriminator
    self.model = set_device(PConvUNet())
    self.optim_args = self.config['optimizer']
    self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
      lr = self.optim_args['lr'])
    self._load()
    if config['distributed']:
      self.model = DDP(self.model, device_ids=[config['global_rank']], output_device=config['global_rank'], 
        broadcast_buffers=True)#, find_unused_parameters=False)
    
    # set summary writer
    self.writer = None
    if self.config['global_rank'] == 0 or (not config['distributed']):
      self.writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))
    self.samples_path = os.path.join(config['save_dir'], 'samples')
    self.results_path = os.path.join(config['save_dir'], 'results')
    
    # other args
    self.log_args = self.config['logger']
    self.train_args = self.config['trainer']

  # get current learning rate
  def get_lr(self):
    return self.optim.param_groups[0]['lr']

  def add_summary(self, writer, name, val):
    if writer is not None and self.iteration % self.log_args['log_step'] == 0:
      writer.add_scalar(name, val, self.iteration)

  def get_non_gan_loss(self, outputs, videos, masks):
    non_gan_losses = []
    b, t, c, h, w = list(outputs.size())
    outputs_feat = self.vgg(outputs.view(b*t, c, h, w))
    videos_feat = self.vgg(videos.view(b*t, c, h, w))
    for loss_name, (loss_instance, loss_weight, input_type) in self.losses.items():
      if loss_weight > 0.0:
        if input_type == 'RGB':
          loss = loss_instance(outputs, videos, masks)
        elif input_type == 'feat':
          loss = loss_instance(outputs_feat, videos_feat, masks)
        loss *= loss_weight
        self.add_summary(self.writer, f'{loss_name}', loss.item())
        non_gan_losses.append(loss)
    return sum(non_gan_losses)


  def train(self):
    total = len(self.train_dataset)
    while True:
      self.epoch += 1
      if self.epoch > self.train_args['epochs']:
        break
      if self.config['distributed']:
        self.train_sampler.set_epoch(self.epoch)
      # self.adjust_learning_rate()
      self._train_epoch()
      # save model and evaluation
      if self.config['global_rank'] == 0:
        if self.epoch % self.train_args['save_freq'] == 0:
          self._save(self.epoch)
          self._eval_epoch(self.epoch)
          print('[**] Training till {} in Rank {}\n'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.config['global_rank']))
    print('\nEnd training....')


  # load netG and netD
  def _load(self):
    model_path = self.config['save_dir']
    if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
      latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
    else:
      ckpts = [os.path.basename(i).split('_netD.pth')[0] for i in glob.glob(os.path.join(model_path, '*_netD.pth'))]
      ckpts.sort()
      latest_epoch = ckpts[-1] if len(ckpts)>0 else None
    if latest_epoch is not None:
      gen_path = os.path.join(model_path, latest_epoch+'_netG.pth')
      if self.config['global_rank'] == 0:
        print('Loading {} generator from {}...'.format(self.config['model']['name'], gen_path))
      data = torch.load(gen_path, map_location = lambda storage, loc: set_device(storage)) 
      model_dict = self.netG.state_dict()
      pretrained_dict = {k:v for k,v in data['netG'].items() if k in model_dict}
      model_dict.update(pretrained_dict)
      self.netG.load_state_dict(model_dict)
      self.optimD.load_state_dict(data['optimD'])
      self.epoch = data['epoch']
      self.iteration = data['iteration']
    else:
      if self.config['global_rank'] == 0:
        print('Warnning: There is no trained model found. An initialized model will be used.')
      
  # save parameters every eval_epoch
  def _save(self, it):
    path = os.path.join(self.config['save_dir'], str(it).zfill(5)+'.pth')
    print('saving {} model to {} ...'.format(self.model_args['name'], path))
    if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.model, DDP):
      netD = self.model.module
    else:
      netG = self.model
    torch.save({'netD': netD.state_dict(), 
      'optimG': self.optimG.state_dict(),
      'optimD': self.optimD.state_dict(),
      'epoch': self.epoch, 'iteration': self.iteration, }, path)
    os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['save_dir'], 'latest.ckpt')))

  # process input and calculate loss every training epoch
  def _train_epoch(self,):
    """
    Training logic for an epoch
    :param epoch: Current epoch number
    """
    raise NotImplementedError

  def _eval_epoch(self,ep):
    """
    Training logic for an epoch
    :param epoch: Current epoch number
    """
    raise NotImplementedError
