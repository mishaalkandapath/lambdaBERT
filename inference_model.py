import matplotlib.pyplot as plt
import os
from models import *
import dataloader
import tqdm, csv, random, pandas as pd
from tokenization import LAMBDA, LAMBDA_LAST, OPEN_RRB, OPEN_RRB_LAST, make_var_emb
from dataloader import BOS_TOKEN, BOS_TOKEN_LAST

import torch 
import torch.nn as nn
import numpy as np

import re, copy
from transformers import BertForMaskedLM, BertConfig
from multipledispatch import dispatch

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random

SEP_ID=102
BOS_ID=101
import time

ALL_VAR_EMBS = make_var_emb(25)

def get_closest_idx(out_vector, in_vectors, in_tokens=None, sep_allowed=True, prejudice_tokens=None):
    #closest euclidean distance:
    if len(out_vector.shape) != 1: out_vector = out_vector.unsqueeze(1)
    vecs = (in_vectors - out_vector)**2
    vecs = vecs.sum(dim=-1)
    best_v_indices = vecs.argsort(dim=-1)

    if in_tokens is not None:
        cur = in_tokens[best_v_indices[0]]
        checker = [SEP_ID]
        while (cur.item() in checker and not sep_allowed) or cur == BOS_ID:
            best_v_indices = best_v_indices[1:]
            cur = in_tokens[best_v_indices[0]]
            if prejudice_tokens: checker = prejudice_tokens
    best_v_index = best_v_indices[0] if len(out_vector.shape) == 1 else best_v_indices[:, 0]
    best_vector = in_vectors[list(range(in_vectors.size(0))), best_v_index]
    return best_v_index

def get_closest_var_idx(out_vector, var_count=0, printing=False):
    #get the closes euclidean distance vector from all vars
    global ALL_VAR_EMBS
    if len(out_vector.shape) != 1: out_vector = out_vector.unsqueeze(1)
    distances = ALL_VAR_EMBS - out_vector.to(ALL_VAR_EMBS.device)
    distances = (distances ** 2).sum(-1)
    if printing: print(distances, distances.argmin(-1), var_count)
    return distances.argmin(-1)

class InferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @dispatch(torch.Tensor, torch.Tensor)
    def forward(self, in_embs, in_tokens, max_len=20, beam_size=0, last=False, bos=None, reference_target=None):
        # if beam_size > 1: 
        #     return self.beam_search_inference(in_embs, in_tokens, max_len=max_len, beam_size=beam_size, last=last, bos=bos)
        if beam_size > 1:
            return self.beam_search_classes(in_embs, 
                                            in_tokens,
                                            max_len=max_len,
                                            beam_size=beam_size,
                                            last=last)
        prs = []
        x = torch.tensor(BOS_TOKEN[os.environ["BERT_TYPE"]] if not last else BOS_TOKEN_LAST[os.environ["BERT_TYPE"]]).to(in_embs.device) if bos is None else bos
        out_stacked, classified_class_stacked, classified_class_stacked_unfiltered, var_reg_stacked = x, torch.tensor([[0]]).long().to(in_embs.device), None, None
        best_index = None
        out_len = 0
        leftover_words=False
        while best_index != in_embs.size(1) and len(prs) < max_len-1 and not leftover_words:
            outs = self.model(x, classified_class_stacked, in_embs, mb_pad=torch.zeros(in_embs.shape[:-1]).to(x.device).to(torch.bool), device=x.device)
            if len(outs) == 3: out, classified_class, var_reg = outs
            else: 
                var_reg = None
                out, classified_class = outs
            prs.append(torch.nn.Softmax()(classified_class.view(-1)).max().item())
            classified_class_ = classified_class.argmax(dim=-1) 
            
            classified_class_stacked_unfiltered = torch.nn.Softmax(dim=-1)(classified_class) if var_reg_stacked is None else torch.cat([classified_class_stacked_unfiltered, torch.nn.Softmax(dim=-1)(classified_class[:, -1]).unsqueeze(1)], dim=1)
            if var_reg is not None: var_reg_stacked = var_reg if var_reg_stacked is None else torch.cat([var_reg_stacked, var_reg[:, -1].unsqueeze(1)], dim=1)
            
            match classified_class_[0, -1].item():
                case 0:
                    to_append = in_embs[:, get_closest_idx(out[0, -1], in_embs[0], in_tokens[0], sep_allowed = leftover_words)]
                case 1:
                    to_append = ALL_VAR_EMBS[get_closest_var_idx(out[0, -1], var_count=torch.count_nonzero(classified_class_[0] == 1))].unsqueeze(0)  
                case 2: 
                    to_append = LAMBDA[os.environ["BERT_TYPE"]] if not last else LAMBDA_LAST[os.environ["BERT_TYPE"]]
                    to_append = torch.tensor(to_append).unsqueeze(0)
                case 3:
                    to_append = OPEN_RRB[os.environ["BERT_TYPE"]] if not last else OPEN_RRB_LAST[os.environ["BERT_TYPE"]]
                    to_append = torch.tensor(to_append).unsqueeze(0)
                case _: 
                    print("UHHH this shudnt happen")
                    to_append = out[:, -1]
            
            to_append = to_append.to(x.device)
            out_stacked = torch.cat([out_stacked, to_append.unsqueeze(1)], dim=1)
            classified_class_stacked = torch.cat([classified_class_stacked, classified_class_[:, -1].unsqueeze(1)], dim=1)

            if classified_class_[0, -1] == 0:
                best_index = in_tokens[0, get_closest_idx(out[0, -1], in_embs[0], in_tokens[0])]
            leftover_words = (torch.count_nonzero(classified_class_ == 0) == in_embs.size(1)-2)
            x = out_stacked

        return out_stacked[:, 1:], classified_class_stacked_unfiltered, var_reg_stacked, torch.tensor(prs)
    
    
    def get_words(self, out, cl, in_embs, in_tokens): #given an output for a sequence, produce the discrete tokenized sequence
        list_out = []
        for i in range(out.shape[0]):
            if cl[i] == 0:
                newest_out = get_closest_idx(out[-1], in_embs, in_tokens, sep_allowed = torch.count_nonzero(cl == 0) == len(in_embs))
                list_out.append(in_tokens[newest_out])
            else:
                list_out.append(101)

        return list_out

    @dispatch(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)   
    def forward(self, seq, seq_syntax, in_embs, mb_pad):
        outs = self.model(seq, seq_syntax, in_embs, mb_pad=mb_pad , device=seq.device)
        return outs
    
    def beam_search_classes(self, in_embs, in_tokens, last=True, beam_size=1, max_len=20):
        self.model.eval()
        with torch.no_grad():
            (out_stacked, 
            classified_class_stacked,
            classified_class_unnormalized) = ((BOS_TOKEN_LAST if last else BOS_TOKEN)[os.environ["BERT_TYPE"]], 
                                torch.tensor([[0]]).long().to(in_embs.device),
                                None)
            out_stacked = torch.tensor(out_stacked).to(in_embs.device)
            out_tokens_stacked = [[BOS_ID]]
            prs = torch.zeros(out_stacked.size(0)).to(in_embs.device)
            beams_left = beam_size
            finished_sentences = []
            finished_prs = []
            finished_classes = []
            while (out_stacked.size(1) < max_len and 
                len(finished_sentences) < beam_size):
                out, classified_class = self.model(out_stacked,
                                                        classified_class_stacked,
                                                        in_embs,
                                                        mb_pad=torch.zeros(in_embs.shape[:-1]).to(out_stacked.device).to(torch.bool))
                classified_class_unnormalized = torch.clone(classified_class)
                classified_class = nn.functional.log_softmax(classified_class,
                                                                dim=-1)
                
                #if we are generating the first token, it is an application
                if out_stacked.size(1) == 1:
                    classified_class_stacked = torch.cat(
                                            [classified_class_stacked, 
                                            torch.tensor([[3]]).to(in_embs.device)
                                            ],
                                            dim=1
                                        )
                    out_stacked = torch.cat([out_stacked, 
                                               torch.tensor((OPEN_RRB_LAST if last else OPEN_RRB)[os.environ["BERT_TYPE"]]).unsqueeze(0).unsqueeze(0).to(in_embs.device)
                                               ],
                                               dim=1
                                            )
                    continue

                # for beams with last prediction == lambda, it must be var next. 
                lambda_batches_indices = torch.where(
                                            classified_class_stacked[:, -1] == 2
                                        )[0].to(in_embs.device)
                #process probability of var
                lambda_classes_batched = classified_class[lambda_batches_indices, -1, 1].to(in_embs.device)
                lambda_classes_class = torch.tensor(
                                                [1]*lambda_batches_indices.size(0)
                                        ).to(in_embs.device)
                                            
                # it can be anything for beams with word, var, or app
                rest_batch_indices = torch.where(
                                                classified_class_stacked[:, -1] != 2)[0].to(in_embs.device)
                rest_classes_batched = classified_class[rest_batch_indices, -1].flatten().to(in_embs.device)
                rest_classes_class = torch.arange(0, 4).repeat(rest_batch_indices.size(0)).to(in_embs.device)
                rest_classes_beam_index = rest_batch_indices.repeat_interleave(4).to(in_embs.device)
                
                #combine probabilities for ranking
                all_classes_batched = torch.cat([rest_classes_batched,
                                                lambda_classes_batched])
                all_classes_class = torch.cat([rest_classes_class,
                                                lambda_classes_class])
                all_classes_beam_index = torch.cat([rest_classes_beam_index,
                                                    lambda_batches_indices])

                # sum
                all_classes_batched += torch.gather(prs,
                                                    0,                   
                                                    all_classes_beam_index)
                # rank
                prs, all_classes_batched = torch.sort(all_classes_batched, stable=True,descending=True)
                prs, all_classes_batched = prs[:beams_left], all_classes_batched[:beams_left]
                all_classes_class = all_classes_class[all_classes_batched]
                all_classes_beam_index = all_classes_beam_index[all_classes_batched]

                #make out stacked, var_reg_stacked, and classified_class_stacked
                classified_class_stacked = torch.cat(
                                        [classified_class_stacked[all_classes_beam_index], all_classes_class.unsqueeze(1)],
                                        dim=1
                                    )
                (word_indices, var_indices,
                lamda_indices, app_indices) = (
                    torch.where(all_classes_class == 0)[0],
                    torch.where(all_classes_class == 1)[0],
                    torch.where(all_classes_class == 2)[0],
                    torch.where(all_classes_class == 3)[0]
                )
                out = out[all_classes_beam_index]
                out_stacked = out_stacked[all_classes_beam_index]
                out_tokens_stacked = [out_tokens_stacked[i] for i in all_classes_beam_index]
                new_out = out[:, -1]
                classified_class_unnormalized = classified_class_unnormalized[all_classes_beam_index]

                
                new_out[lamda_indices] = torch.tensor((LAMBDA_LAST if last else LAMBDA)[os.environ["BERT_TYPE"]]).to(in_embs.device)
                new_out[app_indices] = torch.tensor((OPEN_RRB_LAST if last else OPEN_RRB)[os.environ["BERT_TYPE"]]).to(in_embs.device)

                end_indices = []
                for word_index in word_indices:
                    sep_allowed = len(set(out_tokens_stacked[word_index])) == (in_tokens[0].size(0) - 1)
                    if sep_allowed:
                        end_indices += [word_index]
                        out_tokens_stacked[word_index] += [SEP_ID]
                        continue
                    closest_index = get_closest_idx(
                                                    out[word_index, -1],
                                                    in_embs[0], 
                                                    in_tokens[0],
                                                    sep_allowed=False
                                                )
                    new_out[word_index] = in_embs[
                        0, 
                        closest_index
                    ]
                    out_tokens_stacked[word_index] += [in_tokens[0, closest_index]]
                    
                for var_index in var_indices:
                    new_out[var_index] = ALL_VAR_EMBS[get_closest_var_idx(
                        out[var_index, -1], var_count=torch.count_nonzero(classified_class_stacked[var_index] == 1)
                    )].to(in_embs.device)

                out_stacked = torch.cat([
                                out_stacked, 
                                new_out.unsqueeze(1)],
                                dim=1)

                for end_index in end_indices:
                    finished_sentences.append(out_stacked[end_index, 1:])
                    finished_classes.append(classified_class_stacked[end_index, 1:].to(dtype=torch.int64))
                    finished_prs.append(classified_class_unnormalized[end_index])
                    out_stacked = torch.cat([
                            out_stacked[:end_index],
                            out_stacked[end_index + 1:]
                        ],
                        dim=0)
                    classified_class_stacked = torch.cat([
                            classified_class_stacked[:end_index],
                            classified_class_stacked[end_index + 1:]
                        ],
                        dim=0)
                    prs = torch.cat([
                            prs[:end_index],
                            prs[end_index + 1:]
                        ],
                        dim=0)
                    classified_class_unnormalized = torch.cat([
                            classified_class_unnormalized[:end_index],
                            classified_class_unnormalized[end_index + 1:]
                        ],
                        dim=0)

                    out_tokens_stacked = out_tokens_stacked[:end_index] + out_tokens_stacked[end_index+1:]
                    beams_left-=1
                
                in_embs = in_embs[[0]*beams_left] if in_embs.size(0) != beams_left else in_embs
                classified_class_stacked = classified_class_stacked.to(dtype=torch.int32)
            if beams_left:
                for i in range(beams_left):
                    finished_sentences.append(out_stacked[i, 1:])
                    finished_classes.append(classified_class_stacked[i, 1:].to(dtype=torch.int64))
                    finished_prs.append(classified_class_unnormalized[i])
            return (finished_sentences, finished_classes,
                    None, finished_prs)