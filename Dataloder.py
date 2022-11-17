import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import cv2
from shutil import copy, move
from tqdm.notebook import tqdm
from torchvision import transforms
from torch import random
import torch.utils.data as data
import torch
import torchsummary
import PIL
from PIL import Image, ImageOps

class Cityscapes(data.Dataset):
  def __init__(self, root, IMG_SIZE, phase= 'train', num_classes = 20):
    self.num_classes = num_classes
    self.img_size = IMG_SIZE
    self.root = root
    self.phase = phase
    self.imgs  = list(sorted(os.listdir(os.path.join(root, f"{self.phase}_images", 'images'))))
    self.masks = list(sorted(os.listdir(os.path.join(root, f"{self.phase}_masks", 'masks'))))

    self.input_transform = transforms.Compose([
          transforms.RandomApply(
            torch.nn.ModuleList([
              transforms.RandomRotation(degrees= 10),
              transforms.RandomCrop((self.img_size, self.img_size)),
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
          ])),
          transforms.Resize((self.img_size, self.img_size)),
          transforms.ToTensor(),
          # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
    self.target_transform = transforms.Compose([
          transforms.Grayscale(),
          transforms.PILToTensor(),
          transforms.Lambda(self.label_categorization),
          transforms.RandomApply(
            torch.nn.ModuleList([
              transforms.RandomRotation(degrees= 10),
              transforms.RandomCrop((self.img_size, self.img_size)),
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
          ])),
          transforms.Resize((self.img_size, self.img_size)),
          transforms.Lambda(self.binary_mask),
          # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    print("Total Number of Images:", len(self.imgs))
    print("Total Number of Masks:" , len(self.masks))

  def label_categorization(self, img):
    channled_mask = torch.zeros((self.num_classes, 1024, 2048)) 
    for i in range(self.num_classes):
      # Converting [1, 1024, 2048] to [num_classes, 1024, 2048]
      channled_mask[i][img.squeeze() == i] = 1     
    return channled_mask

  def binary_mask(self, img):
    for i in range(self.num_classes):
     img[i] = (img[i] > 0.50).float() 
    return img
    

  def __getitem__(self, idx):
    img_path  = os.path.join(self.root, f"{self.phase}_images", 'images', self.imgs[idx] )
    mask_path = os.path.join(self.root, f"{self.phase}_masks" , 'masks' , self.masks[idx])

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    seed = np.random.randint(2147483647)
   
    torch.manual_seed(seed) # apply this seed to img transforms
    if self.input_transform is not None:
        img = self.input_transform(img)

    torch.manual_seed(seed) # apply this seed to target transforms
    if self.target_transform is not None:
        mask = self.target_transform(mask)

    return img.type(torch.FloatTensor), mask.type(torch.LongTensor)   

  def __len__(self):
    return self.imgs.__len__()
