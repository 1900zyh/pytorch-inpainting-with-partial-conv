import random
import os 
import numpy as np 
from PIL import Image, ImageFilter
from glob import glob

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from core.utils import ZipReader

try:
  from nvidia.dali.plugin.pytorch import DALIClassificationIterator
  from nvidia.dali.pipeline import Pipeline
  import nvidia.dali.ops as ops
  import nvidia.dali.types as types
except ImportError:
  raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, debug=False, split='train'):
    super(Dataset, self).__init__()
    self.split = split
    self.w, self.h = data_args['w'], data_args['h']
    self.data = [os.path.join(data_args['zip_root'], data_args['name'], i) 
      for i in np.genfromtxt(os.path.join(data_args['flist_root'], data_args['name'], split+'.flist'), dtype=np.str, encoding='utf-8')]
    self.mask = [os.path.join(data_args['zip_root'], 'mask', i)
      for i in np.genfromtxt(os.path.join(data_args['flist_root'], 'mask.flist'), dtype=np.str, encoding='utf-8')]
    self.mask.sort()
    self.data.sort()
    
    self.img_tf = transforms.Compose(
      [transforms.Resize(size=(self.w, self.h)), 
       transforms.ToTensor(),
       transforms.Normalize(mean=MEAN, std=STD)])
    self.mask_tf = transforms.Compose(
      [transforms.Resize(size=(self.w, self.h), interpolation=Image.NEAREST ),
       transforms.ToTensor()])
    if debug:
      self.data = self.data[:100]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      item = self.load_item(0)
    return item

  def load_item(self, index):
    # load image
    img_path = os.path.dirname(self.data[index]) + '.zip'
    img_name = os.path.basename(self.data[index])
    img = ZipReader.imread(img_path, img_name).convert('RGB')
    # load mask 
    m_index = random.randint(0, len(self.mask)-1) if self.split == 'train' else index
    mask_path = os.path.dirname(self.mask[m_index]) + '.zip'
    mask_name = os.path.basename(self.mask[m_index])
    mask = ZipReader.imread(mask_path, mask_name).convert('RGB')
    # augment 
    if self.split == 'train': 
      img = transforms.RandomHorizontalFlip()(img)
      img = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)(img)
      mask = transforms.RandomHorizontalFlip()(mask)
      mask = mask.rotate(random.randint(0,45), expand=True)
      mask = mask.filter(ImageFilter.MaxFilter(3))
    img = self.img_tf(img)
    mask = self.mask_tf(mask)
    return img, 1-mask, img_name

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

def get_dataloader(config, split='train'):
  pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
  pipe.build()
  train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

  pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=crop_size, size=val_size)
  pipe.build()
  val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))