import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def score_group_func(score):
    if score <= 20:
        return 1
    elif score <= 40:
        return 2
    elif score <= 60:
        return 3
    elif score <= 80:
        return 4
    else:
        return 5

dataset_dir = 'petfinder_pawpularity_data'
# Read dataset
data = pd.read_csv(f'{dataset_dir}/train.csv')
# Divide examples into 5 pawpularity score groups: 1, 2, 3, 4, 5.
data['score_group'] = data['Pawpularity'].apply(score_group_func)
# Choose the size of validation set.
validation_size = round(len(data)*0.2, 0)

X = data.drop(['score_group'], axis=1)
y = data['score_group']
nums = [11, 97, 380, 376]
for i in range(4):
    num = nums[i]
    np.random.seed(num)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=num)
    sss.get_n_splits(X, y)
    for train_index, validation_index in sss.split(X, y):
        pass
    train, validation = data.iloc[train_index,:], data.iloc[validation_index,:]

    # Remove column 'score_group' to return the original format of the dataset. 
    train = train.drop(['score_group'], axis=1)
    validation = validation.drop(['score_group'], axis=1)

    # Save to csv files.
    train_name = f'train{i+2}'
    val_name = f'validation{i+2}'
    train.to_csv(f'{dataset_dir}/{train_name}.csv', index=False)
    validation.to_csv(f'{dataset_dir}/{val_name}.csv', index=False)
    print('Done {train_name}, {val_name}')
