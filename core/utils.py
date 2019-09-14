import io 
import numpy as np
import zipfile
from numpy import random 
from PIL import Image 

import torch


# set random seed 
def set_seed(seed, base=0, is_set=True):
  seed += base
  assert seed >=0, '{} >= {}'.format(seed, 0)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, 'r')
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im