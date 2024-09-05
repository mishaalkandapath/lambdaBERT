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
import copy, math

from tokenization import TOKENIZER, BERT_MODEL 

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
            self.initial_forward = nn.Linear(768, d_model)
            self.final_forward = nn.Linear(d_model, 768)
            self.classifier_forward = nn.Linear(d_model, 4)
            self.reg_forward1 = nn.Linear(d_model, d_model)
            self.reg_forward2 = nn.Linear(d_model, 10)
            self.reg_act = nn.ReLU()

            self.decoders = nn.ModuleList([nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout, batch_first=True, norm_first=True)
                                            for _ in range(self.num_layers)])
            
            self.pe_embed = PositionalEncoding(self.d_model)
            
            self.layers_in_a_gpu = self.num_layers

            pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total Trainable Params: {pytorch_total_params}")
                
    def forward(self, seq, emb):
        outputs = seq
        #start from gpu 0
        if self.mode:
            outputs = outputs.to(getattr(self, f"device_{0}"))
            emb = emb.to(getattr(self, f"device_{0}"))

        outputs = self.pe_embed(self.initial_forward(outputs))
        emb = self.pe_embed(self.initial_forward(emb))
        #mask the sequence for autorregressive
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(emb.device)

        final_class_emb_help = None
        #run variable predictions
        # var_outs = self.var_decoder(outputs, emb, tgt_mask=tgt_mask)

        for i in range(self.num_layers):
            outputs = self.decoders[i](outputs, emb, tgt_mask=tgt_mask)

            #change devices if you need to 
            if (i+1) % self.layers_in_a_gpu == 0 and i != self.num_layers - 1 and self.mode:
                outputs = outputs.to(getattr(self, f"device_{i//self.layers_in_a_gpu}"))
                emb = emb.to(getattr(self, f"device_{i//self.layers_in_a_gpu}"))
            if final_class_emb_help is None and i == self.num_layers - 2:
                final_class_emb_help = outputs
            elif self.num_layers-1>i > self.num_layers - 2:
                final_class_emb_help += outputs

        outputs = self.final_forward(outputs)

        #Variable classification and prediction
        classified_class = self.classifier_forward(final_class_emb_help)
        # classified_class = self.classifier_forward(var_outs)

        var_emb = self.reg_forward2(self.reg_act(self.reg_forward1(final_class_emb_help)))
        # var_emb = self.reg_forward2(self.reg_act(self.reg_forward1(var_outs)))

        #dummy
        # var_emb = torch.zeros(outputs.shape[:-1] + (10, ))
        # classified_class = torch.zeros(outputs.shape[:-1] + (3,))
        return outputs, classified_class, var_emb
   
class ShuffledTransformerStack(L.LightningModule):
    def __init__(self, model, t_force=True, t_damp=0.95):
        super().__init__()
        self.model = model
        self.t_force = t_force
        self.t_damp = t_damp if self.t_force else 1
        #pick a random bunch of parameters:
        # self.reference_param = None
        # self.fin_reference = None

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        class_criterion = nn.CrossEntropyLoss()
        in_embs, target_embs, _, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask = batch
        

        with torch.no_grad():
            target_embs, in_embs = target_embs.to(self.device), in_embs.to(self.device)
            # var_index_mask_no = torch.roll(var_index_mask_no, -1, 1) # shift one back coz nps and _ have been moved to the back -- RETIRED: NOW DONE IN process_bert_lambda
            out, classified_class, var_reg = self.model(target_embs[:, :-1, :], in_embs)
            target = target_embs[:, 1:, :]
            
            #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
            #offset the masks by one 
            lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

            return self.common_loss([criterion, class_criterion], [out, classified_class, var_reg], target,
                               lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask, split="valid")

    def common_loss(self, criteria, out, target, lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask, split="train"):
        criterion, class_criterion = criteria
        
        out, classified_class, var_reg = out

        assert len(torch.unique(lambda_index_mask + var_index_mask_no.type(torch.bool) + app_index_mask + pad_mask)) == 2, torch.unique(lambda_index_mask + var_index_mask_no.type(torch.bool) + app_index_mask + pad_mask)
        loss = criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)],
                        target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)])
        

        self.log(f"{split}_loss_tokens", loss, batch_size=out.size(0), sync_dist=True) 
        
        #mse on lambdas
        if out[lambda_index_mask].reshape(-1, out.size(-1)).shape[0] != 0:
            gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) #because lambda's class is 2
            classifier_loss = class_criterion(classified_class.view(-1, 4), gt_cls_mask.view(-1))
            loss += classifier_loss

            self.log(f"{split}_loss_classifier", classifier_loss, batch_size=out.size(0), sync_dist=True)

            #loss on variables: compute the variance on the variables
            var_hot = nn.functional.one_hot(var_index_mask_no.long(), num_classes=torch.unique(var_index_mask_no).size(0))
            var_hot = var_hot.to(dtype=torch.bool)
            out_vars = var_reg.unsqueeze(1) * var_hot.transpose(1, 2).unsqueeze(-1)
           
            # ---- VARIANCE LOSS ----           
            out_var_mean = out_vars.mean(dim=-2, keepdim=True) #* mean_rescale # average on the tokens
            out_var_difference = out_vars - (out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1))
            var_loss = torch.mean(out_var_difference**2, dim=-2, keepdim=True) #* mean_rescale
            var_loss = torch.mean(torch.sum(var_loss.sum(dim=-1).squeeze(-1), dim=-1))
            loss += var_loss

            self.log(f"{split}_loss_variance", var_loss, batch_size=out.size(0), sync_dist=True)

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

        self.log(f"{split}_loss", loss, batch_size=out.size(0), sync_dist=True, prog_bar=True) 

        return loss   

    def training_step(self, batch, batch_idx):
        criterion = nn.MSELoss()

        class_criterion = nn.CrossEntropyLoss()
        (in_embs, target_embs, _, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask) = batch
        
        out = target_embs[:, :-1, :]
        target = target_embs[:, 1:, :]

        #positional embeddings

        out, classified_class, var_reg = self.model(out, in_embs)

        #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
        #offset the masks by one 
        lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])
        loss = self.common_loss([criterion, class_criterion], [out, classified_class, var_reg], target,
                               lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask, split="train")

        # roll out
        out, target = out[:, :-1, :], target[:, 1:, :]
        while (self.t_force and target.size(1) > 2 and (target_embs.size(1) - target.size(1)) <= 10):
            out, classified_class, var_reg = self.model(out, in_embs)
            lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])
            loss += (self.t_damp ** (target_embs.size(1) - target.size(1))) * self.common_loss([criterion, class_criterion], [out, classified_class, var_reg], target,
                               lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask, split="train")
            out = out[:, :-1, :]
            target = target[:, 1:, :]

        return loss
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer
    
class DiscreteTransformerStack(L.LightningModule):
    def __init__(self, model, finetune=False):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.logit_linear = nn.Linear(768, TOKENIZER.vocab_size, bias=False)
        self.b = nn.Parameter(torch.zeros(TOKENIZER.vocab_size))
        self.logit_linear.bias = self.b 
        self.linear = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.LayerNorm(768, 1e-12), self.logit_linear)
        self.finetune = finetune
        
    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        token_criterion = nn.CrossEntropyLoss()
        class_criterion = nn.CrossEntropyLoss()
        in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask = batch
        

        with torch.no_grad():
            target_embs, in_embs = target_embs.to(self.device), in_embs.to(self.device)
            # var_index_mask_no = torch.roll(var_index_mask_no, -1, 1) # shift one back coz nps and _ have been moved to the back -- RETIRED: NOW DONE IN process_bert_lambda
            out, classified_class, var_reg = self.model(target_embs[:, :-1, :], in_embs)
            target = target_embs[:, 1:, :]
            target_tokens = target_tokens[:, 1:]
            
            #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
            #offset the masks by one 
            lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

            target_tokens = target_tokens.to(self.device)
            target_tokens[target_tokens == -1] = 0
            target_tokens = nn.functional.one_hot(target_tokens.long(), num_classes=self.linear[-1].out_features).to(dtype=torch.float32).to(self.device)
            token_out = self.linear(out.detach() if self.finetune else out)
            loss = token_criterion(token_out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)], target_tokens[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)]).mean()
            if not self.finetune: loss += criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)],
                            target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)]).mean()
            
            self.log("val_loss_tokens", loss, batch_size=out.size(0), sync_dist=True)
            if out[lambda_index_mask].reshape(-1, out.size(-1)).shape[0] != 0 and not self.finetune:
                gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool)
                classifier_loss = class_criterion(classified_class.view(-1, 4), gt_cls_mask.view(-1))
                loss += classifier_loss.mean()

                self.log("val_loss_classifier", classifier_loss, batch_size=out.size(0), sync_dist=True)

                #loss on variables: compute the variance on the variables
                var_hot = nn.functional.one_hot(var_index_mask_no.long(), num_classes=torch.unique(var_index_mask_no).size(0))
                # out_vars = out.unsqueeze(1) * var_hot.transpose(1, 2).unsqueeze(-1)
                out_vars = var_reg.unsqueeze(1) * var_hot.transpose(1, 2).unsqueeze(-1)
                #out_vars is of B x VClasses x T x D
                
                # ---- VARIANCE LOSS ----           
                out_var_mean = out_vars.mean(dim=-2, keepdim=True) #* mean_rescale # average on the tokens
                # print(torch.sum(out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1)))
                out_var_difference = out_vars - (out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1))
                # print(out_var_difference.sum())
                var_loss = torch.mean(out_var_difference**2, dim=-2, keepdim=True) #* mean_rescale
                # print(var_loss.sum())
                # var_loss = torch.sqrt(var_loss)
                var_loss = torch.mean(torch.sum(var_loss.sum(dim=-1).squeeze(-1), dim=-1))
                # print(var_loss.sum())
                loss += var_loss.mean()

                self.log("val_loss_variance", var_loss, batch_size=out.size(0), sync_dist=True)

                # --- COS SIM LOSS ----
                out_vars_normed = (out_vars.pow(2).sum(dim=-1) + 1e-6).sqrt()
                out_vars_normed = out_vars / out_vars_normed.unsqueeze(-1)
                out_vars_sim = out_vars_normed @ out_vars_normed.transpose(-1, -2) # similarity within classes, everything else is 0
                out_vars_sim_mask = var_hot.transpose(1, 2).unsqueeze(-1).to(dtype=torch.float32) @ var_hot.transpose(1, 2).unsqueeze(-2).to(dtype=torch.float32)
                var_loss = 1 - (out_vars_sim + (out_vars_sim_mask == 0))
                loss += var_loss.mean()

                self.log("val_loss_vars", var_loss.mean(), batch_size=out.size(0), sync_dist=True)

                # ---- Othogonality loss for Variance Loss
                # ortho_loss = out_var_mean.squeeze(-2) @ out_var_mean.squeeze(-2).transpose(1, 2)
                # #mask diagonals out
                # ortho_loss = ortho_loss * (1 - torch.eye(ortho_loss.size(-1)).to(ortho_loss.device))
                # loss += ortho_loss.mean()

                # --- Ortho loss for sim loss
                out_var_mean_normed = out_vars_normed.mean(dim=-2, keepdim=True)

                # Common component for ortho loss
                ortho_loss = out_var_mean_normed.squeeze(-2) @ out_var_mean_normed.squeeze(-2).transpose(-1, -2)

                #mask diagonals out
                ortho_loss = torch.abs(ortho_loss) * (1 - torch.eye(ortho_loss.size(-1)).to(ortho_loss.device))
                loss += ortho_loss.mean()

                self.log("val_loss_ortho", ortho_loss.mean(), batch_size=out.size(0), sync_dist=True)

                #VAR NORM LOSS
                # norm_loss = 1 - (out_vars.pow(2).sum(dim=-1) + (var_hot.transpose(1, 2).unsqueeze(-1) == 0))
                # loss += norm_loss.mean()

                # self.log("val_norm_loss", norm_loss.mean(), batch_size=out.size(0), sync_dist=True)

        self.log("val_loss", loss.mean(), batch_size=out.size(0), sync_dist=True, prog_bar=True) 

        return loss

    def training_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        token_criterion = nn.CrossEntropyLoss()

        if self.finetune: self.model.eval()

        class_criterion = nn.CrossEntropyLoss()
        (in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask) = batch
        
        out, classified_class, var_reg = self.model(target_embs[:, :-1, :], in_embs)
        target = target_embs[:, 1:, :]
        target_tokens = target_tokens[:, 1:]

        #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
        #offset the masks by one 
        lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

        assert len(torch.unique(lambda_index_mask + var_index_mask_no.type(torch.bool) + app_index_mask + pad_mask)) == 2, torch.unique(lambda_index_mask + var_index_mask_no.type(torch.bool) + app_index_mask + pad_mask)
        # loss = criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)],
        #                 target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)])
        
        target_tokens = target_tokens.to(self.device)
        target_tokens[target_tokens == -1] = 0
        target_tokens = nn.functional.one_hot(target_tokens.long(), num_classes=self.linear[-1].out_features).to(dtype=torch.float32).to(self.device)
        
        token_out = self.linear(out.detach() if self.finetune else out)
        loss = token_criterion((token_out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)]), target_tokens[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)]).mean()
        if not self.finetune: 
            loss += criterion(out[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)],
                            target[~(lambda_index_mask | var_index_mask_no.type(torch.bool) | app_index_mask | pad_mask)]).mean()
            
        self.log("train_loss_tokens", loss, batch_size=out.size(0), sync_dist=True)
        if out[lambda_index_mask].reshape(-1, out.size(-1)).shape[0] != 0 and not self.finetune:
            # lambda_loss = criterion(out[lambda_index_mask].reshape(-1, out.size(-1)), target[lambda_index_mask].reshape(-1, out.size(-1))) # has to be consisten across batches

            ##inconsisten version:
            # lambda_loss = criterion(out[lambda_index_mask].reshape(-1, out.size(-1))[0:1, :].repeat(repeat_times, 1), out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :])
            # loss += lambda_loss.mean()

            # self.log("train_loss_lambdas", lambda_loss, batch_size=out.size(0), sync_dist=True) 
            gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) #because lambda's class is 2
            classifier_loss = class_criterion(classified_class.view(-1, 4), gt_cls_mask.view(-1))
            loss += classifier_loss.mean()

            self.log("train_loss_classifier", classifier_loss, batch_size=out.size(0), sync_dist=True)

            #loss on variables: compute the variance on the variables
            var_hot = nn.functional.one_hot(var_index_mask_no.long(), num_classes=torch.unique(var_index_mask_no).size(0))
            # out_vars = out.unsqueeze(1) * var_hot.transpose(1, 2).unsqueeze(-1) -- FOR PREVIOUS
            var_hot = var_hot.to(dtype=torch.bool)
            out_vars = var_reg.unsqueeze(1) * var_hot.transpose(1, 2).unsqueeze(-1)
           
            # ---- VARIANCE LOSS ----           
            # mean_rescale = (out_vars.shape[-2]/torch.count_nonzero(out_vars.sum(dim=-1, keepdim=True), dim=-2)).unsqueeze(-1).detach()
            out_var_mean = out_vars.mean(dim=-2, keepdim=True) #* mean_rescale # average on the tokens
            # print(torch.sum(out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1)))
            out_var_difference = out_vars - (out_var_mean * var_hot.transpose(1, 2).unsqueeze(-1))
            # print(out_var_difference.sum())
            var_loss = torch.mean(out_var_difference**2, dim=-2, keepdim=True) #* mean_rescale
            # print(var_loss.sum())
            # var_loss = torch.sqrt(var_loss)
            var_loss = torch.mean(torch.sum(var_loss.sum(dim=-1).squeeze(-1), dim=-1))
            # print(var_loss.sum())
            loss += var_loss.mean()

            self.log("train_loss_variance", var_loss, batch_size=out.size(0), sync_dist=True)

            # --- COS SIM LOSS ----
            out_vars_normed = (out_vars.pow(2).sum(dim=-1) + 1e-6).sqrt()
            out_vars_normed = out_vars / out_vars_normed.unsqueeze(-1)
            out_vars_sim = out_vars_normed @ out_vars_normed.transpose(-1, -2) # similarity within classes, everything else is 0
            out_vars_sim_mask = var_hot.transpose(1, 2).unsqueeze(-1).to(dtype=torch.float32) @ var_hot.transpose(1, 2).unsqueeze(-2).to(dtype=torch.float32)
            var_loss = 1 - (out_vars_sim + (out_vars_sim_mask == 0))
            loss += var_loss.mean()

            self.log("train_loss_vars", var_loss.mean(), batch_size=out.size(0), sync_dist=True)
            
            # ---- Othogonality loss for Variance Loss
            # out_var_mean_normed = torch.sum((out_var_mean ** 2), dim=-1, keepdim=True) + 1e-6
            # out_var_mean_normed = out_var_mean / out_var_mean_normed

            # --- Ortho loss for sim loss
            out_var_mean_normed = out_vars_normed.mean(dim=-2, keepdim=True)

            # Common component for ortho loss
            ortho_loss = out_var_mean_normed.squeeze(-2) @ out_var_mean_normed.squeeze(-2).transpose(-1, -2)

            #mask diagonals out
            ortho_loss = torch.abs(ortho_loss) * (1 - torch.eye(ortho_loss.size(-1)).to(ortho_loss.device))
            loss += ortho_loss.mean()

            self.log("train_loss_ortho", ortho_loss.mean(), batch_size=out.size(0), sync_dist=True)

            #sample a few variable vectors from a random batch and write them to file
            # if batch_idx % 100 == 0:
            #     batch = torch.randint(0, 4)
            #     #pick a random variable in this batch

            #     vrs = out_vars[, ]

            #VAR NORM LOSS
            # norm_loss = 1 - (out_vars.pow(2).sum(dim=-1) + (var_hot.transpose(1, 2).unsqueeze(-1) == 0))
            # loss += norm_loss.mean()

            # self.log("train_norm_loss", norm_loss.mean(), batch_size=out.size(0), sync_dist=True)
            self.log("train_var_norm", torch.mean((out_vars.pow(2).sum(dim=-1) + 1e-6).sqrt()))

        self.log("train_loss", loss, batch_size=out.size(0), sync_dist=True, prog_bar=True) 

        return loss   

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
def load_model(path=None):
    model = TransformerDecoderStack(4, 384, 8, 3072)
    if path:
        checkpoint = torch.load(args.model_path)
        model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(model_weights)
    return model

def main(hparams=None, load_chckpnt=False, discrete=False, finetune=False, **kwargs):
    
    if load_chckpnt: model = load_model(load_chckpnt)
    else: model = TransformerDecoderStack(4, 384, 8, 3072)
    if discrete: 
        model = DiscreteTransformerStack(model, finetune=finetune)
    else: 
        wrapper = ShuffledTransformerStack
        model = wrapper(model, t_force = kwargs["t_force"], t_damp=kwargs["t_damp"])

    logger = WandbLogger(log_model="all", project="lambdaBERT", entity="mishaalkandapath") #CSVLogger(SAVE_DIR+"logs_after_5/")
    checkpointing = L.pytorch.callbacks.ModelCheckpoint(dirpath=SAVE_DIR,
        filename='constrative_{epoch}',
        save_top_k=-1,
        every_n_epochs=4,
        save_on_train_epoch_end=True)
    trainer = L.Trainer(max_epochs=200, callbacks=[checkpointing], log_every_n_steps=1, num_sanity_val_steps=0, logger=logger, default_root_dir=SAVE_DIR+"models/")
    train_dataloader, val_dataloader = dataloader.data_init(kwargs["batch_size"], last=kwargs["bert_is_last"])
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
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--finetune_discrete", action="store_true")
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--t_damp", default=0.95, type=float)
    parser.add_argument("--t_force", action="store_true")
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--batch_size", default=50, type=int)
    parsear.add_argument("--bert_is_last", action="store_true")


    args = parser.parse_args()
    SAVE_DIR = args.save_dir
    main(load_chckpnt=args.model_path, discrete=args.discrete, finetune=args.finetune_discrete, t_force=args.t_force, t_damp=args.t_damp, batch_size=args.batch_size, bert_is_last=args.bert_is_last)


    # model = TransformerDecoderStack(6, 384, 12, 3072)
    # print("-- Initialized Model --")
    # print("Dataloading...")
    # train_dataloader, val_dataloader, test_dataloader = dataloader.data_init()
    # print("--Training Begins--")
    # train(model, train_dataloader, val_dataloader, 10)
