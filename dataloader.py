import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

import random
import pandas as pd

#create a directory where the key is a csv. each row has first column as the raw text sentence, and the second col being the 
# path to the file that stores all its lambda terms


class LambdaTermsDataset(Dataset):
    def __init__(self, input_sentences_file, main_dir, transform=None, target_transform=None):
        self.main_dir = main_dir
        self.input_sentences = pd.read_csv(input_sentences_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.input_sentences)

    def __getitem__(self, index):
        sentence = self.input_sentences.iloc[index, 0]
        path = self.input_sentences.iloc[index, 1]
        with open(path, 'r') as f:
            lambda_terms = f.readlines()

        if self.transform:
            sentence = self.transform(sentence)
        if self.target_transform:
            lambda_terms = self.target_transform(lambda_terms)
        
        return sentence, lambda_terms
    

def data_init():
    
    dataset = LambdaTermsDataset('data/input_sentences.csv', 'data/lambda_terms', shuffle=True)
    #split the datset 70 20 10 split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader

            