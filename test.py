# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import datetime
import random
import sys
import json
import glob

### My libs
from core.utils import set_device, postprocess, ZipReader, set_seed
from core.utils import postprocess, unnormalize
from core.model import PConvUNet
from core.dataset import Dataset
 

parser = argparse.ArgumentParser(description="PConv")
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-p", "--port", type=str, default="23451")
args = parser.parse_args()

BATCH_SIZE = 4

def main_worker(gpu, ngpus_per_node, config):
  if config['distributed']:
    torch.cuda.set_device(int(config['local_rank']))
    print('using GPU {} for training'.format(int(config['local_rank'])))
    torch.distributed.init_process_group(backend = 'nccl', 
      init_method = config['init_method'],
      world_size = config['world_size'], 
      rank = config['global_rank'],
      group_name='mtorch'
    )
  set_seed(config['seed'])

  # Model and version
  model = set_device(PConvUNet())
  latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  path = os.path.join(config['save_dir'], latest_epoch+'.pth')
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  model.load_state_dict(data['model'])
  model = DDP(model, device_ids=[gpu], output_device=gpu, broadcast_buffers=True)
  model.eval() 

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test')
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['data_loader']['num_workers'], pin_memory=True)

  
  path = os.path.join(config['save_dir'], 'results_{}'.format(str(latest_epoch).zfill(5)))
  os.makedirs(path, exist_ok=True)
  # iteration through datasets
  for idx, (images, masks, names) in enumerate(dataloader):
    print('[{}] {}/{}: {}  ...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
      idx*images.size(0), len(dataloader), names[0]))
    inpts = images*masks
    images, inpts, masks = set_device([images, inpts, masks])
    output, _ = model(inpts, masks)
    orig_imgs = list(postprocess(images))
    comp_imgs = list(postprocess(masks*images+(1-masks)*output))
    for i in range(len(orig_imgs)):
      Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_orig.png'.format(names[i].split('.')[0])))
      Image.fromarray(comp_imgs[i]).save(os.path.join(path, '{}_comp.png'.format(names[i].split('.')[0])))
  print('Finish in {}'.format(path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  config['save_dir'] = os.path.join(config['save_dir'], config['data_loader']['name'])

  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
 