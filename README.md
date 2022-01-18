## PET FINDER PAWPULARITY CONTEST

### 1. Introduction
Petfinder is a contest hosted by [PetFinder.my](https://www.petfinder.my) on [Kaggle](https://www.kaggle.com/c/petfinder-pawpularity-score). 

The purpose of this competition is to analyze raw images and photo metadata to predict the “Pawpularity” of pet photos. The Pawpularity Score is calculated from the number of view statistics of each pet's profile at the listing pages. Through the predicted score of each pet photo, PetFinder.my can instruct rescuers how to improve the photo quality, which helps increasing the opportunities for a pet to be adopted faster. 

![Example](https://github.com/uyenhnp/hateful_memes_challenge/blob/master/demo/theme.jpg)

### 2. Data
The official web page for the dataset: [link](https://www.kaggle.com/c/petfinder-pawpularity-score/data).

The training dataset includes 9912 photos with the corresponding photo metadata. For more information about photo metadata, please review this [link](https://www.kaggle.com/c/petfinder-pawpularity-score/data?select=train). 

### 3. Models

| Model | Description | CV RMSE |
| ----------- | ----------- | ----------- |
| Baseline | ResNet50 | 17.940 |
| Baseline | swin_base_patch4_window12_384 | 17.4658 |
| Animal Aware Model |  swin_base_patch4_window12_384 + CLIP | 17.4592 |

From the final leaderboard of the competition, the test RMSE of Animal Aware Model is 17.31516 (the test set comprises about 6800 images).  

### 4. Usage
To use the Animal Aware Model, please folow the code below. 

#### Reproduce the model
If you want to reproduce the model, please download the dataset and put the training images into the folder `petfinder_pawpularity_data/train` (the location of the images should be as follows: `petfinder_pawpularity_data/train/{img_name}.jpg`), then run the following code:
```sh
python train.py --model aam --backbone swin_base_patch4_window12_384 --train_full --use_clip --use_animal_aware_embedding --epoch_decay 7
```

#### Use the model
Please download the checkpoint via this [link](https://drive.google.com/file/d/1qXwO5J-sKHB8Xtz8OR6TmnLiJDHXTGs3/view?usp=sharing), then put the file into the folder `checkpoints` and run the following code:  

```sh
import torch
from torch.utils.data import DataLoader
from models.animal_aware_model import AnimalAwareModel
from memes_dataset import PetDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the checkpoint
model = AnimalAwareModel(backbone='swin_base_patch4_window12_384', drop=0.5, loss='BCE', use_embedding=True)
model.to(device)
checkpoint = torch.load('checkpoints/aam.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# - or - 
model.train()
```