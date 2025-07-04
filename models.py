import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tokenization
import dataloader

import os
import torch

import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import argparse
import copy, math
import time

from transformers import BertConfig, BertForMaskedLM

SAVE_DIR = "/home/mishaalk/scratch/lmabdaModelNoTForce/"

### Distributed Training Modules ###
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

GLOBAL_BIG_FILE = open("big_vectors.txt", "a")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

### Model Definition ###
class TransformerDecoderStack(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.2, devices=[], custom=False):
        super(TransformerDecoderStack, self).__init__()
        self.custom = custom
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.initial_forward_out = nn.Linear(768, d_model)
        self.initial_forward_emb = nn.Linear(768, d_model)
        self.final_forward = nn.Linear(d_model, 768)
        self.classifier_forward = nn.Linear(d_model, 4)
        
        if not self.custom:
            self.reg_forward1 = nn.Linear(d_model, d_model)
            self.reg_forward2 = nn.Linear(d_model, 100)
            self.reg_act = nn.GELU()

        self.decoders = nn.ModuleList([nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout, batch_first=True, norm_first=True)
                                        for _ in range(self.num_layers)])
        
        self.pe_embed = PositionalEncoding(self.d_model)
        self.syntax_embed = nn.Embedding(4, self.d_model)
        
        self.layers_in_a_gpu = self.num_layers

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Trainable Params: {pytorch_total_params}")
                
    def forward(self, seq, seq_syntax, emb, mb_pad=None, device="cuda"):

        outputs = seq
        outputs = self.pe_embed(self.initial_forward_out(outputs)) # does th add in the pemebed forward
        outputs += self.syntax_embed(seq_syntax)

        emb = self.initial_forward_emb(emb)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(emb.device)

        emb *= (~mb_pad.unsqueeze(-1))

        for i in range(self.num_layers):
            outputs = self.decoders[i](outputs, emb, tgt_mask=tgt_mask, tgt_is_causal=True)
        
        #Variable classification and prediction
        classified_class = self.classifier_forward(outputs) # predict the classifier absed on this 
        if not self.custom: var_emb = self.reg_forward2(self.reg_act(self.reg_forward1(outputs)))

        outputs = self.final_forward(outputs) # back to 768

        # var_emb = torch.zeros_like(outputs.sum(-1).unsqueeze(-1)) --- debugging without zes
        if self.custom:
            return outputs, classified_class
        return outputs, classified_class, var_emb
   
class ShuffledTransformerStack(L.LightningModule):
    def __init__(self, model, t_force=True, t_damp=0.95, custom=False):
        super().__init__()
        self.model = model
        self.t_force = t_force
        self.t_damp = t_damp if self.t_force else 1
        self.custom = custom

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        # class_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,8]).to(self.device).to(dtype=torch.float))
        class_criterion = nn.CrossEntropyLoss()
        in_embs, target_embs, _, var_index_mask_no, lambda_index_mask, app_index_mask, stop_mask, pad_mask, sent_pad_mask = batch
        

        with torch.no_grad():
            target_embs, in_embs, sent_pad_mask = target_embs.to(self.device), in_embs.to(self.device), sent_pad_mask.to(self.device)

            seq_syntax = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool)
            
            o = self.model(target_embs[:, :-1, :], seq_syntax[:, :-1], in_embs, sent_pad_mask, self.device)
            
            if not self.custom: out, classified_class, var_reg = o
            else: out, classified_class = o

            target = target_embs[:, 1:, :]
            
            #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
            #offset the masks by one 
            lambda_index_mask, app_index_mask, var_index_mask_no, stop_mask, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], stop_mask[:, 1:], pad_mask[:, 1:])

            out_list = [out, classified_class, var_reg] if not self.custom else [out, classified_class]
            return self.common_loss([criterion, class_criterion], out_list, target,
                               lambda_index_mask, app_index_mask, var_index_mask_no, stop_mask, pad_mask, split="valid", 
                               bos=None, in_embs=in_embs, sent_pad_mask=sent_pad_mask)

    def common_loss(self, criteria, out, target, lambda_index_mask, app_index_mask, var_index_mask_no, stop_mask, pad_mask, split="train",
                     bos=None, in_embs=None, sent_pad_mask=None):
        criterion, class_criterion = criteria
        
        if not self.custom: out, classified_class, var_reg = out
        else: out, classified_class = out
        filter_mask = ~(var_index_mask_no.type(torch.bool) | pad_mask)
        token_loss = criterion(out[filter_mask],
                        target[filter_mask]) # use pads because pads are stops
        loss = token_loss
        # normed_vector = lambda x : x/torch.linalg.vector_norm(x, dim=-1, ord=2, keepdim=True)
        # normed_loss = criterion(normed_vector(out[filter_mask]), 
        #                                       normed_vector(target[filter_mask]))
        # self.log(f"{split}_loss_normed_tokens", normed_loss, batch_size=out.size(0), sync_dist=True)
        
        #mse on lambdas
        if out[lambda_index_mask].reshape(-1, out.size(-1)).shape[0] != 0:
            # print("tf " ,F.softmax(classified_class, dim=-1)[0, :20])
            gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) #+ 4*stop_mask.type(torch.bool) #because lambda's class is 2
            classifier_loss = class_criterion(classified_class.view(-1, 4), gt_cls_mask.view(-1))
            loss += classifier_loss

            if self.custom:
                var_loss = criterion(out[(var_index_mask_no.type(torch.bool))],
                        target[(var_index_mask_no.type(torch.bool))]) # use pads because pads are stops
                loss += var_loss
            else:
                #loss on variables: compute the variance on the variables
                var_hot = nn.functional.one_hot(var_index_mask_no.long(), num_classes=torch.unique(var_index_mask_no).size(0)) #batch x length x vars
                var_hot = var_hot.to(dtype=torch.bool)
                out_vars = var_reg.unsqueeze(1) * var_hot.transpose(1, 2).unsqueeze(-1) #batch x 1 x length x 10 * batch x vars x length x 1
                #giving batch x vars x length x 10
            
                # ---- VARIANCE LOSS ----           
                out_var_mean = out_vars.mean(dim=-2, keepdim=True) #* mean_rescale # average on the tokens
                out_var_difference = out_vars - (out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1))
                var_loss = torch.mean(out_var_difference**2, dim=-2, keepdim=True) #* mean_rescale
                var_loss = torch.mean(torch.sum(var_loss.sum(dim=-1).squeeze(-1), dim=-1))
                loss += var_loss

                self.log(f"{split}_loss_variance", var_loss, batch_size=out.size(0), sync_dist=True)

                #together with other losses, the addition of below becomes linearity loss:
                #the sum of all of them is equal to the mean of them 
                #--- LINEARITY LOSS ---
                linearity_loss = (2*out_var_mean.squeeze(-2) - torch.sum(out_vars, dim=-2))**2 #batch x vars x 10
                loss += torch.mean(torch.sum(linearity_loss.sum(dim=-1), dim=-1))

                # --- COS SIM LOSS ----
                out_vars_normed = (out_vars.pow(2).sum(dim=-1) + 1e-6).sqrt()
                out_vars_normed = out_vars / out_vars_normed.unsqueeze(-1)
                out_vars_sim = out_vars_normed @ out_vars_normed.transpose(-1, -2) # similarity within classes, everything else is 0
                out_vars_sim_mask = var_hot.transpose(1, 2).unsqueeze(-1).to(dtype=torch.float32) @ var_hot.transpose(1, 2).unsqueeze(-2).to(dtype=torch.float32)
                var_loss = 1 - (out_vars_sim + (out_vars_sim_mask == 0))
                loss += var_loss.mean()

                self.log(f"{split}_loss_vars", var_loss.mean(), batch_size=out.size(0), sync_dist=True)

                # --- Ortho loss for sim loss
                out_var_mean_normed = out_vars_normed.mean(dim=-2, keepdim=True)

                # Common component for ortho loss
                ortho_loss = out_var_mean_normed.squeeze(-2) @ out_var_mean_normed.squeeze(-2).transpose(-1, -2)

                #mask diagonals out
                ortho_loss = torch.abs(ortho_loss) * (1 - torch.eye(ortho_loss.size(-1)).to(ortho_loss.device))
                loss += ortho_loss.mean()

                self.log(f"{split}_loss_ortho", ortho_loss.mean(), batch_size=out.size(0), sync_dist=True)

                self.log(f"{split}_var_norm", torch.mean((out_vars.pow(2).sum(dim=-1) + 1e-6).sqrt()), sync_dist=True)

        self.log_dict({f"{split}_loss_tokens": token_loss, f"{split}_loss_classifier": classifier_loss, f"{split}_loss_var_reg": var_loss.mean(), f"{split}_loss": loss}, batch_size=out.size(0), sync_dist=True, prog_bar=True)
        # self.log(, prog_bar=True, batch_size=out.size(0), sync_dist=True)
        return loss   

    def training_step(self, batch, batch_idx):
        criterion = nn.MSELoss()

        # class_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,8]).to(self.device).to(dtype=torch.float))
        class_criterion = nn.CrossEntropyLoss()
        (in_embs, target_embs, _, var_index_mask_no, lambda_index_mask, app_index_mask, stop_mask, pad_mask, sent_pad_mask) = batch
        
        out = target_embs[:, :-1, :]
        target = target_embs[:, 1:, :]

        seq_syntax = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool)
        o = self.model(out, seq_syntax[:, :-1], in_embs, sent_pad_mask, self.device)

        if not self.custom: out, classified_class, var_reg = o
        else: out, classified_class = o

        #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
        #offset the masks by one 
        lambda_index_mask, app_index_mask, var_index_mask_no, stop_mask, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], stop_mask[:, 1:], pad_mask[:, 1:])
        out_list = [out, classified_class, var_reg] if not self.custom else [out, classified_class]
        loss = self.common_loss([criterion, class_criterion], out_list, target,
                               lambda_index_mask, app_index_mask, var_index_mask_no, stop_mask, pad_mask, split="train",bos=None, in_embs=in_embs, sent_pad_mask=sent_pad_mask)

        return loss
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
def load_model(path=None, custom=False):
    model = TransformerDecoderStack(4, 384, 8, 3072, custom=custom)
    if path:
        checkpoint = torch.load(args.model_path)
        model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(model_weights)
    return model

def main(hparams=None, load_chckpnt=False, **kwargs):
    
    if load_chckpnt: model = load_model(load_chckpnt, custom=kwargs["custom_t"])
    else: model = TransformerDecoderStack(4, 384, 8, 3072, custom=kwargs["custom_t"])

    model = ShuffledTransformerStack(model, t_force = kwargs["t_force"], t_damp=kwargs["t_damp"], custom=kwargs["custom_t"])

    logger = WandbLogger(log_model="all", project="lambdaBERT", entity="mishaalkandapath") #CSVLogger(SAVE_DIR+"logs_after_5/")
    checkpointing = L.pytorch.callbacks.ModelCheckpoint(dirpath=SAVE_DIR,
        filename='constrative_{epoch}',
        save_top_k=-1,
        every_n_epochs=4,
        save_on_train_epoch_end=True)
    
    # profiler = AdvancedProfiler(filename="out_log.txt")
    trainer = L.Trainer(max_epochs=120, callbacks=[checkpointing], log_every_n_steps=1, num_sanity_val_steps=0, logger=logger, default_root_dir=SAVE_DIR+"models/")#, profiler=profiler)
    train_dataloader, val_dataloader, test_dataloader = dataloader.data_init(kwargs["batch_size"], last=kwargs["bert_is_last"], rem_spec_sentences=kwargs["rem_spec_sentences"], data_path=kwargs["data_path"])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    #make arg parser
    #set visible gpus:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    L.seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=384, help="Model Dimension")
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--t_damp", default=0.95, type=float)
    parser.add_argument("--t_force", action="store_true")
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--custom_transformer", action="store_true")
    parser.add_argument("--bert_is_last", action="store_true")
    parser.add_argument("--rem_spec_sentences", action="store_true")
    parser.add_argument("--data_path", default="") # can also be base


    args = parser.parse_args()
    SAVE_DIR = args.save_dir
    torch.manual_seed(0)
    #assert (args.model_is_discrete and args.model_path) or (args.model_path) or not (args.model_is_discrete or args.model_path), "model path for discrete model to be provided"
    main(load_chckpnt=args.model_path, t_force=args.t_force,
         t_damp=args.t_damp, batch_size=args.batch_size, 
         custom_t=args.custom_transformer,
         bert_is_last=args.bert_is_last, 
         rem_spec_sentences=args.rem_spec_sentences, 
         data_path=args.data_path)


    # model = TransformerDecoderStack(6, 384, 12, 3072)
    # print("-- Initialized Model --")
    # print("Dataloading...")
    # train_dataloader, val_dataloader, test_dataloader = dataloader.data_init()
    # print("--Training Begins--")
    # train(model, train_dataloader, val_dataloader, 10)
