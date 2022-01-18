from os import replace
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


def prepare_meta_data(element):
    if element == 0:
        return -1
    else:
        return element

class PetDataset(Dataset):
    def __init__(self, dataset_dir='petfinder_pawpularity_data', 
                 split='train', debug_small=False, crop=(0.2, 1), img_size=384,
                 loss = 'BCE', feat_engineer=False, pawscore_jitter=0,
                 grayscale=False, use_clip=False, preprocess_clip=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.loss = loss
        self.feat_engineer = feat_engineer
        self.pawscore_jitter = pawscore_jitter
        self.use_clip = use_clip

        # Read metadata file. 
        data = pd.read_csv(f'{dataset_dir}/{split}.csv')
        if debug_small:
            data = data.sample(n=160, replace=False, axis=0)
        self.data = data

        if split == 'train' or split == 'train1' or split == 'train2' or \
                split == 'train3' or split == 'train4' or split == 'train5':
            if grayscale:
                self.img_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(size=(img_size, img_size), scale=crop, interpolation=3),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.img_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(size=(img_size, img_size), scale=crop, interpolation=3),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.img_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        if use_clip:
            self.preprocess_clip = preprocess_clip

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, index):
        example = self.data.iloc[index,:]
        id = example['Id']

        # Open an image.
        if self.split == 'test':
            img_folder = 'test'
        else:
            img_folder = 'train'
        img = Image.open(f'{self.dataset_dir}/{img_folder}/{id}.jpg').convert(mode='RGB')
        img_tensor = self.img_transforms(img)

        if self.use_clip:
            img_clip = self.preprocess_clip(img)

        # Get metadata.
        if self.split == 'test':
            meta = example[1:] # remove Id
        else:
            meta = example[1:-1] # remove Id & Pawpularity columns
        meta = meta.apply(prepare_meta_data) # convert label 0 to -1
        if self.feat_engineer:
            meta = meta[['Occlusion', 'Blur']].values
        else:
            meta = meta.values
        meta = meta.astype(float).copy()
        meta_tensor = torch.from_numpy(meta).float()

        # Blur label.
        blur_label = example[['Blur']].values.astype(int).copy()
        blur_label_tensor = torch.from_numpy(blur_label).long().squeeze()

        # Get the pawpularity score.
        if self.split != 'test':
            if self.loss == 'MSE':
                score = example['Pawpularity']
            elif self.loss == 'BCE':
                score = example['Pawpularity']
                if 'train' in self.split and self.pawscore_jitter > 0:
                    score = self.jitter_paw_score(score)
                score /= 100
            score_tensor = torch.FloatTensor([score])
            out = {
                'img': img_tensor,
                'meta': meta_tensor,
                'blur': blur_label_tensor,
                'score': score_tensor
            }
        else:
            out = {
                'img': img_tensor,
                'meta': meta_tensor,
                'blur': blur_label_tensor,
            }

        if self.use_clip:
            out['img_clip'] = img_clip
        
        return out

    def jitter_paw_score(self, score):
        v_min = max(1, score - self.pawscore_jitter)
        v_max = min(100, score + self.pawscore_jitter)
        new_score = np.random.randint(v_min, v_max + 1)
        return new_score