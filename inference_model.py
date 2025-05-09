import matplotlib.pyplot as plt
from models import *
import dataloader
import tqdm, csv, random, pandas as pd
from tokenization import get_bert_emb, TOKENIZER, LAMBDA, OPEN_RRB, BERT_MODEL
from dataloader import SEP_TOKEN, BOS_TOKEN, BOS_TOKEN_LAST

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

def get_closest_idx(out_vector, in_vectors):
    #closest euclidean distance:
    vecs = (in_vectors - out_vector.unsqueeze(1))**2
    vecs = vecs.sum(dim=-1)
    best_v_index = vecs.argmin(dim=-1)
    best_vector = in_vectors[list(range(in_vectors.size(0))), best_v_index]
    return best_vector

class InferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @dispatch(torch.Tensor)
    def forward(self, in_embs, max_len=20, classify=True, beam_size=0, last=False, unfiltered_class=False, reference=None):
        # NOTE: classify shud be deprecated
        if beam_size >= 1: 
            return self.beam_search_inference(in_embs, max_len=max_len, classify=classify, beam_size=beam_size, reference=reference)
        pr = 1
        x = torch.tensor(BOS_TOKEN if not last else BOS_TOKEN_LAST).to(in_embs.device)
        out_stacked, classified_class_stacked, classified_class_stacked_unfiltered, var_reg_stacked, newest_out = x, torch.tensor([[0]]).long().to(in_embs.device) if classify else None, None, None, None
        list_out = []
        while newest_out != 102 and len(list_out) < max_len-1:
            out, classified_class, var_reg = self.model(x, classified_class_stacked, in_embs, mb_pad=torch.zeros(in_embs.shape[:-1]).to(x.device).to(torch.bool), device=x.device)
            if classify: 
                pr *= torch.nn.Softmax()(classified_class.view(-1)).max().item()
                classified_class_ = classified_class.argmax(dim=-1) # CHANGE IF INCLUDING STOP
            if var_reg_stacked is None:
                var_reg_stacked = var_reg
                classified_class_stacked_unfiltered = classified_class
                out_stacked = torch.cat([out_stacked, find_best_out(out[:, -1], in_embs)], dim=1)
                classified_class_stacked = torch.cat([classified_class_stacked, classified_class_[:, -1].unsqueeze(1)], dim=1)
            else:
                out_stacked = torch.cat([out_stacked, find_best_out(out[:, -1], in_embs)], dim=1)
                classified_class_stacked = torch.cat([classified_class_stacked, classified_class_[:, -1].unsqueeze(1)], dim=1)
                var_reg_stacked = torch.cat([var_reg_stacked, var_reg[:, -1].unsqueeze(1)], dim=1)
                classified_class_stacked_unfiltered = torch.cat([classified_class_stacked_unfiltered, classified_class[:, -1].unsqueeze(1)], dim=1)
            

            if not classify: cls_probs = nn.functional.softmax(classified_class[-1, -1], dim=-1)
            # print(cls_probs.sort(-1))
            if classified_class_[-1, -1] == 0:
                sorted_stiff, newest_out = self.linear(out[0, -1, :]).sort(-1)
                # print(TOKENIZER.convert_ids_to_tokens(newest_out[-5:]))
                # print(nn.functional.softmax(sorted_stiff, dim=-1)[-5:])
                #decode and print the top 5:

                newest_out = newest_out[-1].item()
                list_out.append(newest_out)
            else:
                list_out.append(101)
            # print("->EOW->")
            x = out_stacked
        return torch.tensor(list_out).unsqueeze(0) if not classify else torch.tensor(list_out), classified_class_stacked_unfiltered if unfiltered_class else torch.nn.functional.one_hot(classified_class_stacked[:, 1:], num_classes=4), var_reg_stacked, pr
    
    
    def get_words(self, out, cl): #given an output for a sequence, produce the discrete tokenized sequence
        list_out = []
        for i in range(out.shape[0]):
            if cl[i] == 0:
                sorted_stiff, newest_out = self.linear(out[i, :]).sort(-1)
                newest_out = newest_out[-1].item()
                list_out.append(newest_out)
            else:
                list_out.append(101)

        return list_out

    def beam_search_inference(self, in_embs, max_len=20, classify=True, beam_size=1, reference=None): # TODO: make beam start from 5 -- number of classes
        self.model.eval()
        x = BOS_TOKEN_LAST#torch.tensor(BOS_TOKEN_LAST).to(in_embs.device)
        out_stacked, classified_class_stacked, var_reg_stacked, newest_out = x, torch.tensor([[0]]).long().to(x.device) if classify else None, None, None
        list_out = []
        cls_out = []
        var_out = []
        prob_list = []
        probs_list = []
        beam_size =1

        while out_stacked.size(1) < max_len:
            # process each beam 
            out, classified_class, var_reg = self.model(x, classified_class_stacked, in_embs, mb_pad=torch.zeros(in_embs.shape[:-1]).to(x.device).to(torch.bool), device=x.device)
            print(nn.functional.softmax(classified_class, dim=-1))
            if out_stacked.size(1) > 11:
                raise Exception
            if classify: 
                # did we have a lambda last time?
                # lmda_mask = torch.where(classified_class_stacked[:, -1] == 2)
                # classified_class[lmda_mask[0], -1, 0] = -torch.inf # cant be a word
                # classified_class[lmda_mask[0], -1, 2:] = -torch.inf
                # classified_class[lmda_mask[0], -1, 2] += 1e-4 # small boost

                #sort for beams
                classified_class = nn.functional.softmax(classified_class, dim=-1)
                cls_probs, classified_class = classified_class.sort(-1, descending=True) # batch x length x 4
                #classified_class = classified_class[:, :, :beam_size] # batch x length x beam
                #cls_probs = cls_probs[:, :, :beam_size] # batch x length x beam
                
                #make beams
                if prob_list == []: #first time
                    classified_class = classified_class[:, :, :beam_size].squeeze(0).T # beam x length -- because batch is 1
                    cls_probs = cls_probs[:, :, :beam_size].squeeze(0).T # similarly 
                    prob_list.extend(sum(cls_probs.tolist(), start=[])) #get the different beam probabilities -- not gonna log coz it might be to small or smth
                    probs_list.extend(sum(cls_probs.tolist(), start=[]))
                    probs_list = torch.tensor(probs_list).to(classified_class.device).unsqueeze(1)
                    b_indices = [0]*min(beam_size, 4)
                else:
                    # get the beam best from each, flatten, take beam best 
                    #beam x length x 1 * batch x beam x length
                    #prod the prbabilities to get the sequences probability so far
                    cl_og = cls_probs
                    cls_probs, classified_class_ = (torch.tensor(prob_list).to(x.device).unsqueeze(-1) * cls_probs.transpose(-1, -2)[:, : , -1]).flatten().sort(descending=True)               
                    cl_og = cl_og.transpose(-1, -2)[:, : , -1].flatten()[classified_class_]
                    #at this point i have sorted for each of the previous beams, its best succeeding beams. 
                    classified_class_ = classified_class_[:beam_size]
                    b_indices = classified_class_//classified_class.size(-1) #which previous beam is it from?
                    cl_indices = classified_class_ % classified_class.size(-1) #which new beam is it?

                    cls_probs = cls_probs[:beam_size]
                    cl_og = cl_og[:beam_size]

                    #classified class needs to be beam x length
                    classified_class = classified_class[b_indices, :, cl_indices]

                    #modify prob_list
                    prob_list = torch.tensor(prob_list).to(b_indices.device)[b_indices] if type(prob_list) is list else prob_list[b_indices]
                    probs_list = probs_list[b_indices]
                    probs_list = torch.cat([probs_list, cl_og.unsqueeze(1)], dim=-1)
                    prob_list *= cls_probs
                    # print(prob_list, cls_probs)
            
            in_embs = torch.repeat_interleave(in_embs[0:1], len(b_indices), dim=0).to(x.device)
            notword_indices = torch.where(classified_class[:, -1] != 0) # classified_class is batch x length, giving not_word indices size of batch
            out = out[b_indices]
            word_replaced = find_best_out(out[:, -1], in_embs)
            word_replaced[notword_indices] = out[notword_indices[0], -1]
            
            if out_stacked.size(1) > 2:
                print(reference[0, out_stacked.size(1)-1])
                print(((reference[0, out_stacked.size(1)-1] - out[0, -1])**2).sum(dim=-1))
                print(word_replaced, notword_indices)
                time.sleep(60)
            if var_reg_stacked is None:
                var_reg_stacked = var_reg
                out_stacked = torch.cat([out_stacked[b_indices], word_replaced.unsqueeze(1)], dim=1) 
                var_reg_stacked = var_reg_stacked.squeeze(0).repeat(min(beam_size, 4), 1, 1)
                classified_class_stacked = classified_class_stacked.squeeze(0).repeat(min(beam_size, 4), 1)
                classified_class_stacked = torch.cat([classified_class_stacked, classified_class[:, -1].unsqueeze(1)], dim=1)
                
            else:
                out_stacked = torch.cat([out_stacked[b_indices], word_replaced.unsqueeze(1)], dim=1) # get the outputs of these batches only
                var_reg_stacked = torch.cat([var_reg_stacked[b_indices], var_reg[b_indices, -1].unsqueeze(1)], dim=1)
                classified_class_stacked = torch.cat([classified_class_stacked[b_indices], classified_class[:, -1].unsqueeze(1)], dim=1)           

            if not classify: cls_probs = nn.functional.softmax(classified_class[-1, -1], dim=-1)
            
            # check if theres a sentence that is to be stopped
            rem_list = []
            not_rem = []
            for i in range(classified_class_stacked.size(0)):
                if (classified_class_stacked[i, -1] == 0): # if we have a word
                    sorted_stiff, newest_out = self.linear(out_stacked[i, -1, :]).sort(-1)
                    newest_out = newest_out[-1].item()
                else: newest_out = None
                if newest_out == 102 or (out_stacked.size(1) == max_len and len(list_out) != beam_size): 
                    # print(TOKENIZER.convert_ids_to_tokens([newest_out]))
                    if len(list_out) < beam_size or prob_list[i] > list_out[0][0]:
                        if len(list_out) == beam_size: list_out = list_out[1:]
                        list_out.append((prob_list[i], random.random(), probs_list[i].detach().cpu(), self.get_words(out_stacked[i], classified_class_stacked[i]), 
                                                                    classified_class_stacked[i, 1:],
                                                                    var_reg_stacked[i])) # save the sequence
                        list_out.sort() # maintain ordering
                        rem_list.append(i)
                        continue
                not_rem.append(i)
                assert len(list_out) <= beam_size, len(list_out)
            if len(rem_list) == beam_size: break
            
            for i in rem_list:
                probs_list[i] = probs_list[not_rem[0]]
                prob_list[i] = prob_list[not_rem[0]]
                out_stacked[i] = out_stacked[not_rem[0]]
                var_reg_stacked[i] = var_reg[not_rem[0]]
                classified_class_stacked[i] = classified_class_stacked[not_rem[0]]

            x = out_stacked
            assert len(list_out) <= beam_size, len(list_out)
    
        assert len(list_out) <= beam_size, len(list_out)
        list_out.sort(key=lambda x: x[0], reverse=True)
        ps, _, pss, list_out, cls_out, var_out = (zip(*list_out))   
        # print(pss)
        return list_out, cls_out, var_out, ps, pss

    @dispatch(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)   
    def forward(self, seq, seq_syntax, in_embs, mb_pad):
        outs, classified_class, var_emb = self.model(seq, seq_syntax, in_embs, mb_pad=mb_pad , device=seq.device)
        return outs, classified_class, var_emb