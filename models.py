import torch 
import torch.nn as nn
import torch.optim as optim

import tokenization
import dataloader

import os
import torch

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import argparse
import copy

SAVE_DIR = "/w/150/lambda_squad/lambdaBERT/save/"

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
        #pick a random bunch of parameters:
        # self.reference_param = None
        # self.fin_reference = None

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        (seq, target_seq) = batch
        

        with torch.no_grad():
            tokenized, in_pad_mask = tokenization.create_out_embeddings(seq)
            target_tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = tokenization.create_out_embeddings(target_seq, lamda=True)

            tokenization.BERT_MODEL.to(self.device)
            tokenized.to(self.device)

            in_embs = tokenization.get_bert_emb(tokenized)
            in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

            target_tokenized.to(self.device)
            target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, (var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask), lambda_norm=True, var_norm=True)
            
            target_embs, in_embs = target_embs.to(self.device), in_embs.to(self.device)
            lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask.to(self.device), var_index_mask.to(self.device), var_index_mask_underscore.to(self.device), var_index_mask_no.to(self.device), pad_mask.to(self.device))


            out = self.model(target_embs[:, :-1, :], in_embs)
            target = target_embs[:, 1:, :]
            
            #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
            #offset the masks by one 
            lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], var_index_mask[:, 1:], var_index_mask_underscore[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

            loss = criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)],
                            target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)])

        # self.log("val_loss_tokens", loss, batch_size=out.size(0), sync_dist=True) 
        
        # #take all lambdas into account and compute their difference with the first lambda
        # repeat_times = out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :].shape[0]
        # lambda_loss = criterion(out[lambda_index_mask].reshape(-1, out.size(-1))[0:1, :].repeat(repeat_times, 1), out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :])
        # loss += lambda_loss

        # self.log("val_loss_lambdas", lambda_loss, batch_size=out.size(0), sync_dist=True) 
        
        # #contrastive losses on the variables at some point?
        # #get the variables only :
        # out_vars, target_vars = out[var_index_mask_no.type(torch.bool)], target[var_index_mask_no.type(torch.bool)]

        # #get the first indices of the variables involved. 
        # flattened_var_mask = var_index_mask_no[var_index_mask_no != 0]
        # _, reverse_indices, counts = torch.unique(flattened_var_mask, return_inverse=True, return_counts=True, sorted=True)
        # ind_sorted = torch.argsort(reverse_indices.to(torch.uint8), stable=True)
        # ind_sorted = ind_sorted.to(self.device)
        # cum_sum = counts.cumsum(0) - 1
        # cum_sum = torch.cat([torch.tensor([0]).to(self.device), cum_sum])
        # first_indices = ind_sorted[cum_sum]
        
        # #mseloss on this
        # target_vars = out_vars[first_indices][reverse_indices] #reference embeddings arrangement
        # correct_nos = torch.count_nonzero(torch.count_nonzero(out_vars - target_vars, axis=-1) == 0)

        # #make sure on gpu
        # var_loss = criterion(out_vars, target_vars.detach())
        # loss += var_loss
        # self.log("val_loss_vars", var_loss, batch_size=out.size(0), sync_dist=True) 
        
        # #count var var mismatches
        # out_vars = out_vars.unsqueeze(1) - out_vars.unsqueeze(0)
        # #count the number of zeros here
        # nos = torch.count_nonzero(torch.count_nonzero(out_vars, axis=-1) == 0) - correct_nos
        # loss += nos
        # self.log("val_count_loss", nos.to(dtype=torch.float32), batch_size=out.size(0), sync_dist=True) 

        # self.log("val_loss", loss, batch_size=out.size(0), sync_dist=True, prog_bar=True) 
        self.log("val_loss", loss.mean(), batch_size=out.size(0), sync_dist=True, prog_bar=True) 

        return loss

    def training_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        (seq, target_seq) = batch

        #tokenize the sqequences
        tokenized, in_pad_mask = tokenization.create_out_embeddings(seq)
        target_tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = tokenization.create_out_embeddings(target_seq, lamda=True)

        tokenization.BERT_MODEL.to(self.device)
        tokenized.to(self.device)
        #get the bert embeddings
        in_embs = tokenization.get_bert_emb(tokenized)
        #mask out the pads:
        in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

        #get the bert embeddings for the target sequence
        target_tokenized.to(self.device)
        lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask.to(self.device), var_index_mask.to(self.device), var_index_mask_underscore.to(self.device), var_index_mask_no.to(self.device), pad_mask.to(self.device))
        target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, (var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask), lambda_norm=True, var_norm=True)
        var_index_mask_no = torch.roll(var_index_mask_no, -1, 1) # shift one back coz nps and _ have been moved to the back
        tokenization.BERT_MODEL.to('cpu')
        
        out = self.model(target_embs[:, :-1, :], in_embs)
        target = target_embs[:, 1:, :]

        #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
        #offset the masks by one 
        lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], var_index_mask[:, 1:], var_index_mask_underscore[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

        loss = criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)],
                        target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | pad_mask)])

        self.log("train_loss_tokens", loss, batch_size=out.size(0), sync_dist=True) 
        
        #mse on lambdas
        if out[lambda_index_mask].reshape(-1, out.size(-1)).shape[0] != 0:
            lambda_loss = criterion(out[lambda_index_mask].reshape(-1, out.size(-1)), target[lambda_index_mask].reshape(-1, out.size(-1))) # has to be consisten across batches

            ##inconsisten version:
            # lambda_loss = criterion(out[lambda_index_mask].reshape(-1, out.size(-1))[0:1, :].repeat(repeat_times, 1), out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :])
            loss += lambda_loss.mean()

            self.log("train_loss_lambdas", lambda_loss, batch_size=out.size(0), sync_dist=True) 

            #loss on variables: compute the variance on the variables
            var_hot = nn.functional.one_hot(var_index_mask_no.long(), num_classes=torch.unique(var_index_mask_no).size(0))
            out_vars = out.unsqueeze(1) * var_hot.transpose(1, 2).unsqueeze(-1)
            mean_rescale = (out_vars.shape[-2]/torch.count_nonzero(out_vars.sum(dim=-1, keepdim=True), dim=-2)).unsqueeze(-1).detach()
            out_var_mean = out_vars.mean(dim=-2, keepdim=True) #* mean_rescale # average on the tokens
            # print(torch.sum(out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1)))
            out_var_difference = out_vars - (out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1))
            # print(out_var_difference.sum())
            var_loss = torch.mean(out_var_difference**2, dim=-2, keepdim=True) #* mean_rescale
            # print(var_loss.sum())
            # var_loss = torch.sqrt(var_loss)
            var_loss = torch.mean(torch.sum(var_loss.sum(dim=-1).squeeze(-1), dim=-1))
            # print(var_loss.sum())
            loss += var_loss

            self.log("train_loss_vars", var_loss.mean(), batch_size=out.size(0), sync_dist=True)
        self.log("train_loss", loss, batch_size=out.size(0), sync_dist=True, prog_bar=True) 

        return loss   

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def main(hparams=None, load_chckpnt=False):
    model = TransformerDecoderStack(4, 384, 8, 3072)
    if load_chckpnt: 
        model = LitTransformerStack.load_from_checkpoint(SAVE_DIR+"logs/lightning_logs/version_0/checkpoints/epoch=4-step=485.ckpt",model=model)
        print("sucesfully loaded in parameters")
    else: model = LitTransformerStack(model)

    logger = WandbLogger(log_model="all", project="lambdaBERT", entity="mishaalkandapath") #CSVLogger(SAVE_DIR+"logs_after_5/")
    trainer = L.Trainer(max_epochs=5, num_sanity_val_steps=0, logger=logger, default_root_dir=SAVE_DIR+"models/")
    train_dataloader, val_dataloader = dataloader.data_init(10)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    #make arg parser
    #set visible gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    L.seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=384, help="Model Dimension")

    args = parser.parse_args()

    main(load_chckpnt=False)


    # model = TransformerDecoderStack(6, 384, 12, 3072)
    # print("-- Initialized Model --")
    # print("Dataloading...")
    # train_dataloader, val_dataloader, test_dataloader = dataloader.data_init()
    # print("--Training Begins--")
    # train(model, train_dataloader, val_dataloader, 10)
