import torch 
import torch.nn as nn
import torch.optim as optim

import tokenization
import dataloader

import os
import torch

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import argparse

SAVE_DIR = "/home/mishaalk/scratch/data/"

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
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.2, mode=0, devices=[]):
        super(TransformerDecoderStack, self).__init__()
        self.mode = mode
        if mode:
            assert len(devices) % 2 == 0, "Number of devices must be even"
            # make device atts
            for i, device in enumerate(devices):
                assert torch.cuda.is_available(device), f"Device {device} is not available"
                setattr(self, f"device_{i}", device)

            self.d_model = d_model
            self.num_layers = num_layers
            self.initial_forward = nn.Linear(768, d_model, device = getattr(self, f"device_{0}"))
            self.final_forward = nn.Linear(d_model, 768, device=getattr(self, f"device_{len(devices) - 1}"))

            self.decoders = nn.ModuleList([nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout, device=device, batch_first=True)
                                            for _ in range(self.num_layers)])

            self.layers_in_a_gpu = num_layers // len(devices)
            for i in range(self.num_layers): # send decoders to the appropriate devices
                self.decoders[i] = self.decoders[i].to(getattr(self, f"device_{i//self.layers_in_a_gpu}"))

            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total Trainable Params: {pytorch_total_params}")
        else:
            self.d_model = d_model
            self.num_layers = num_layers
            self.initial_forward = nn.Linear(768, d_model).cuda()
            self.final_forward = nn.Linear(d_model, 768).cuda()

            self.decoders = nn.ModuleList([nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout, batch_first=True).cuda()
                                            for _ in range(self.num_layers)])
            
            self.layers_in_a_gpu = self.num_layers

            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total Trainable Params: {pytorch_total_params}")
                
    def forward(self, seq, emb):
        outputs = seq
        #start from gpu 0
        if self.mode:
            outputs = outputs.to(getattr(self, f"device_{0}"))
            emb = emb.to(getattr(self, f"device_{0}"))

        outputs = self.initial_forward(outputs)
        emb = self.initial_forward(emb)
        #mask the sequence of autoregressivity
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(emb.device)

        for i in range(self.num_layers):
            outputs = self.decoders[i](outputs, emb, tgt_mask=tgt_mask)

            #change devices if you need to 
            if (i+1) % self.layers_in_a_gpu == 0 and i != self.num_layers - 1 and self.mode:
                outputs = outputs.to(getattr(self, f"device_{i//self.layers_in_a_gpu}"))
                emb = emb.to(getattr(self, f"device_{i//self.layers_in_a_gpu}"))

        outputs = self.final_forward(outputs)
        return outputs

class LitTransformerStack(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        (seq, target_seq) = batch
        tokenized, in_pad_mask = tokenization.create_out_embeddings(seq)
        target_tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = tokenization.create_out_embeddings(target_seq, lamda=True)

        in_embs = tokenization.get_bert_emb(tokenized)
        in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

        target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, (var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask), lambda_norm=True, var_norm=True)
        
        target_embs, in_embs = target_embs.to(self.device), in_embs.to(self.device)
        lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask.to(self.device), var_index_mask.to(self.device), var_index_mask_underscore.to(self.device), var_index_mask_no.to(self.device), pad_mask.to(self.device))
        
        out = self.model(target_embs[:, -1, :], in_embs)
        target = target_embs[:, 1:, :]
        loss = criterion(out, target)

        self.log("val_loss", loss)

    def training_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        (seq, target_seq) = batch

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
        target_embs, in_embs = target_embs.to(self.device), in_embs.to(self.device)
        lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask.to(self.device), var_index_mask.to(self.device), var_index_mask_underscore.to(self.device), var_index_mask_no.to(self.device), pad_mask.to(self.device))
        out = self.model(target_embs[:, :-1, :], in_embs)
        target = target_embs[:, 1:, :]

        #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
        #offset the masks by one 
        lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], var_index_mask[:, 1:], var_index_mask_underscore[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

        loss = criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)],
                        target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)])
        
        #take all lambdas into account and compute their difference with the first lambda
        repeat_times = out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :].shape[0]
        loss += criterion(out[lambda_index_mask].reshape(-1, out.size(-1))[0:1, :].repeat(repeat_times, 1), out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :])

        #contrastive losses on the variables at some point?
        #get the variables only :
        out_vars, target_vars = out[var_index_mask_no.type(torch.bool)], target[var_index_mask_no.type(torch.bool)]

        #get the first indices of the variables involved. 
        flattened_var_mask = var_index_mask_no[var_index_mask_no != 0]
        _, reverse_indices, counts = torch.unique(flattened_var_mask, return_inverse=True, return_counts=True, sorted=True)
        ind_sorted = torch.argsort(reverse_indices.to(torch.uint8), stable=True)
        ind_sorted = ind_sorted.to(self.device)
        cum_sum = counts.cumsum(0) - 1
        cum_sum = torch.cat([torch.tensor([0]).to(self.device), cum_sum])
        first_indices = ind_sorted[cum_sum]
        
        #mseloss on this
        target_vars = out_vars[first_indices][reverse_indices] #reference embeddings arrangement
        correct_nos = torch.count_nonzero(torch.count_nonzero(out_vars - target_vars, axis=-1) == 0)

        #make sure on gpu
        loss += criterion(out_vars, target_vars.detach())
        
        #count var var mismatches
        out_vars = out_vars.unsqueeze(1) - out_vars.unsqueeze(0)
        #count the number of zeros here
        nos = torch.count_nonzero(torch.count_nonzero(out_vars, axis=-1) == 0) - correct_nos
        loss += nos

        return loss   

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train(model, train_loader, val_loader, epochs, lr=0.001, rank=0):

    with torch.device(rank):
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
                in_embs = tokenization.get_bert_emb(tokenized, rank)
                #mask out the pads:
                in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

                #get the bert embeddings for the target sequence
                target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, (var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask), rank, lambda_norm=True, var_norm=True)
                #in_embs will be our encoder embs
                out = model(target_embs[:, :-1, :], in_embs)
                target = target_embs[:, 1:, :]

                #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
                #offset the masks by one 
                lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], var_index_mask[:, 1:], var_index_mask_underscore[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

                loss = criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)],
                                target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)])
                
                #take all lambdas into account and compute their difference with the first lambda
                repeat_times = out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :].shape[0]
                loss += criterion(out[lambda_index_mask].reshape(-1, out.size(-1))[0:1, :].repeat(repeat_times, 1), out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :])

                #contrastive losses on the variables at some point?
                #get the variables only :
                out_vars, target_vars = out[var_index_mask_no.type(torch.bool)], target[var_index_mask_no.type(torch.bool)]

                #get the first indices of the variables involved. 
                flattened_var_mask = var_index_mask_no[var_index_mask_no != 0]
                _, reverse_indices, counts = torch.unique(flattened_var_mask, return_inverse=True, return_counts=True, sorted=True)
                ind_sorted = torch.argsort(reverse_indices.to(torch.uint8), stable=True)
                cum_sum = counts.cumsum(0) - 1
                cum_sum = torch.cat([torch.tensor([0]), cum_sum])
                first_indices = ind_sorted[cum_sum]
                
                #mseloss on this
                target_vars = out_vars[first_indices][reverse_indices] #reference embeddings arrangement
                correct_nos = torch.count_nonzero(torch.count_nonzero(out_vars - target_vars, axis=-1) == 0)

                #make sure on gpu
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
    model = TransformerDecoderStack(4, 384, 8, 3072).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    train_dataloader, val_dataloader = dataloader.data_init(100)

    train(ddp_model, train_dataloader, val_dataloader, 10, rank=rank)
    
    cleanup()

def ddp_mp_training(rank, world_size):
    setup(rank, world_size)
    model = TransformerDecoderStack(4, 768, 8, 3072, [i*2 +1 for i in range(rank)]) # alternate the gpus
    ddp_model = DDP(model)
    
    #checkpointing
    CHECKPOINT_PATH = SAVE_DIR + "models/model.checkpoint"
    if rank == 0:
        torch.save(model.state_dict(), CHECKPOINT_PATH)
    
    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} #move from 0 to current rank
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    train_dataloader, val_dataloader = dataloader.data_init(100)
    train(ddp_model, train_dataloader, val_dataloader, 10)

    cleanup()

def main(hparams=None):
    model = TransformerDecoderStack(4, 384, 8, 3072)
    model = LitTransformerStack(model)

    logger = CSVLogger(SAVE_DIR+"logs/")
    trainer = L.Trainer(max_epochs=5, logger=logger, default_root_dir=SAVE_DIR+"models/")
    train_dataloader, val_dataloader = dataloader.data_init(100)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    #make arg parser
    #set visible gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0, help="0 for ddp, 1 for ddp with model parallelism")
    parser.add_argument("--d_model", type=int, default=384, help="Model Dimension")

    args = parser.parse_args()
    
    if args.mode == 0:
        mp.spawn(ddp_training, args=(2, ), nprocs=2, join=True) # world_size = 2
    elif args.mode == 1:
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus//2
        mp.spawn(ddp_mp_training,
                args=(world_size,),
                nprocs=world_size,
                join=True)
    else:
        main()


    # model = TransformerDecoderStack(6, 384, 12, 3072)
    # print("-- Initialized Model --")
    # print("Dataloading...")
    # train_dataloader, val_dataloader, test_dataloader = dataloader.data_init()
    # print("--Training Begins--")
    # train(model, train_dataloader, val_dataloader, 10)