import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

import random
import pandas as pd

from tokenization import TOKENIZER, BERT_MODEL, create_out_tensor

#create a directory where the key is a csv. each row has first column as the raw text sentence, and the second col being the 
# path to the file that stores all its lambda terms

DATA_PATH = "/w/150/lambda_squad/lambdaBERT/data/"


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
        if sentence[0] == '""': sentence = sentence[1]
        if sentence[-1] == '""': sentence = sentence[:-1]
        path = self.input_sentences.iloc[index, 2]
        path = DATA_PATH + path[len("lambdaBERT/data/"):]
        with open(path, 'r') as f:
            lambda_terms = f.readlines()[0].strip()

        # remove the ")" from the lambda_term:
        lambda_terms = lambda_terms.replace(")", "")

        if self.transform:
            sentence = self.transform(sentence)
        if self.target_transform:
            lambda_terms = self.target_transform(lambda_terms)
        
        return sentence, lambda_terms
    
class ShuffledLambdaTermsDataset(Dataset):
    def __init__(self, input_sentences_file, main_dir, transform=None, target_transform=None):
        self.main_dir = main_dir
        self.input_sentences = pd.read_csv(input_sentences_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.input_sentences)

    def __getitem__(self, index):
        sentence = self.input_sentences.iloc[index, 1]
        if sentence[0] == '""': sentence = sentence[1]
        if sentence[-1] == '""': sentence = sentence[:-1]
        path = self.input_sentences.iloc[index, 2]
        path = DATA_PATH + path[len("lambdaBERT/data/"):]
        with open(path, 'r') as f:
            lambda_terms = f.readlines()[0].strip()

        # remove the ")" from the lambda_term:
        lambda_terms = lambda_terms.replace(")", "")

        target_embs, lambda_index_mask, var_index_mask_no, app_index_mask = create_out_tensor(sentence, lambda_terms)
        
        return sentence, lambda_terms

def shuffled_collate(batch):
    sent_embedding, lambda_term_embedding, var_mask, lambda_mask, app_mask = batch
    sent_embedding_batched = torch.nn.rnn.pack_sequence(sent_embedding, batch_first=True, padding_value = 15)
    lambda_term_embedding_batched = torch.nn.rnn.pack_sequence(lambda_term_embedding, batch_first=True, padding_value = 15)
    var_mask_batched = torch.nn.rnn.pack_sequence(var_mask, batch_first=True, padding_value = 0)
    lambda_mask_batched = torch.nn.rnn.pack_sequence(lambda_mask, batch_first=True, padding_value = 0)
    app_mask_batched = torch.nn.rnn.pack_sequence(app_mask, batch_first=True, padding_value = 0)

    sent_pad_mask = sent_embedding_batched != 15
    lambda_pad_mask = lambda_term_embedding_batched != 15

    return sent_embedding_batched, lambda_term_embedding_batched, var_mask_batched, lambda_mask_batched, app_mask_batched, sent_pad_mask, lambda_pad_mask


def data_init(batch_size, test=False, shuffled=False):
    
    #load in the tokenizer
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if not shuffled: dataset = LambdaTermsDataset(DATA_PATH + 'input_sentences.csv', DATA_PATH + 'lambda_terms/')
    else: dataset = ShuffledLambdaTermsDataset(DATA_PATH + 'input_sentences.csv', DATA_PATH + 'lambda_terms/')
    #split the datset 70 20 10 split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    if test:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=shuffled_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=shuffled_collate)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=shuffled_collate)
        return train_dataloader, val_dataloader, test_dataloader
    else:
        train_size += test_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=9, collate_fn=shuffled_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=9, collate_fn=shuffled_collate)

        return train_dataloader, val_dataloader



            
