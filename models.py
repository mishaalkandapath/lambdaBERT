import torch 
import torch.nn as nn
import torch.optim as optim

import tokenization
import dataloader

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

### Distributed Training Modules ###
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


### Model Definition ###
class TransformerDecoderStack(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.2, device="cpu"):
        super(TransformerDecoderStack, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout, device=device, batch_first=True)
        self.decoders = nn.TransformerDecoder(decoder_layer, num_layers)
        self.d_model = d_model
        self.num_layers = num_layers
        self.final_forward = nn.Linear(d_model, d_model, device=device)
    
    def forward(self, seq, emb):
        out = self.decoders(seq, emb)
        return out

def train(model, train_loader, val_loader, epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        bar = tqdm(enumerate(train_loader), unit="batch", total=len(train_loader))
        for i, (seq, target_seq) in bar:
            #tokenize the sqequences
            tokenized, in_pad_mask = tokenization.create_out_embeddings(seq)
            target_tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = tokenization.create_out_embeddings(target_seq, lamda=True)

            #get the bert embeddings
            in_embs = tokenization.get_bert_emb(tokenized)
            #mask out the pads:
            in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

            #get the bert embeddings for the target sequence
            target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, (var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask), lambda_norm=True, var_norm=True)
            #in_embs will be our encoder embs
            out = model(target_embs[:, :-1, :], in_embs)
            target = target_embs[:, 1:, :]

            #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
            #offset the masks by one 
            lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], var_index_mask[:, 1:], var_index_mask_underscore[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])
            
            loss = criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)],
                            target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)])
            
            #take all lambdas into account and compute their difference with the first lambda #TODO: if this doesnt work, try a random perm pairing difference
            repeat_times = out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :].shape[0]
            loss += criterion(out[lambda_index_mask].reshape(-1, out.size(-1))[0:1, :].repeat(repeat_times, 1), out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :])

            #contrastive losses on the variables at some point?
            #get the variables only :
            out_vars, target_vars = out[var_index_mask_no.type(torch.bool)], target[var_index_mask_no.type(torch.bool)]

            #get the first indices of the variables involved. 
            flattened_var_mask = var_index_mask_no[var_index_mask_no != 0]
            _, reverse_indices, counts = torch.unique(flattened_var_mask, return_inverse=True, return_counts=True, sorted=True)
            ind_sorted = torch.argsort(reverse_indices, stable=True)
            cum_sum = counts.cumsum(0) - 1
            cum_sum = torch.cat([torch.tensor([0]), cum_sum])
            first_indices = ind_sorted[cum_sum]
            
            #mseloss on this
            target_vars = out_vars[first_indices][reverse_indices] #reference embeddings arrangement
            correct_nos = torch.count_nonzero(torch.count_nonzero(out_vars - target_vars, axis=-1) == 0)
            loss += criterion(out_vars, target_vars.detach())
            
            #count var var mismatches
            out_vars = out_vars.unsqueeze(1) - out_vars.unsqueeze(0)
            #count the number of zeros here
            nos = torch.count_nonzero(torch.count_nonzero(out_vars, axis=-1) == 0) - correct_nos
            loss += nos

            if i % 100 == 0:
                bar.set_description(f"Epoch {epoch} Iteration {i} Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for i, (seq, target_seq) in enumerate(val_loader):
                tokenized, in_pad_mask = tokenization.create_out_embeddings(seq)
                target_tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = tokenization.create_out_embeddings(target_seq, lamda=True)

                in_embs = tokenization.get_bert_emb(tokenized)
                in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

                target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, var_index_mask, lambda_norm=True, var_norm=True)
                out = model(target_embs[:, -1, :], in_embs)
                target = target_embs[:, 1:, :]
                loss = criterion(out, target)

                bar.set_description(f"Epoch {epoch} Validation Loss: {loss.item()}")
    return model

def ddp_training(rank, world_size):
    setup(rank, world_size)
    model = TransformerDecoderStack(4, 768, 8, 3072).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train_dataloader, val_dataloader, test_dataloader = dataloader.data_init(100)

    train(ddp_model, train_dataloader, val_dataloader, 10)
    
    cleanup()



if __name__ == "__main__":

    mp.spawn(ddp_training, args=(2, ), nprocs=2, join=True) # world_size = 2
    # model = TransformerDecoderStack(6, 768, 12, 3072)
    # print("-- Initialized Model --")
    # print("Dataloading...")
    # train_dataloader, val_dataloader, test_dataloader = dataloader.data_init()
    # print("--Training Begins--")
    # train(model, train_dataloader, val_dataloader, 10)