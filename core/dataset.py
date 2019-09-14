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
    img = img.resize((self.w, self.h), Image.ANTIALIAS)
    # load mask 
    m_index = random.randint(0, len(self.mask)) if self.split == 'train' else index
    mask_path = os.path.dirname(self.mask[m_index]) + '.zip'
    mask_name = os.path.basename(self.mask[m_index])
    mask = ZipReader.imread(mask_path, mask_name).convert('L')
    mask = mask.resize((self.w, self.h), Image.ANTIALIAS)
    # augment 
    if self.split == 'train': 
      img = transforms.RandomHorizontalFlip()(img)
      img = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)(img)
      mask = transforms.RandomHorizontalFlip()(mask)
      mask = mask.rotate(random.randint(0,45), expand=True)
      mask = filter(ImageFilter.MaxFilter(np.randint(2,5)))
    return F.to_tensor(img)*2-1., F.to_tensor(mask), img_name

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item
