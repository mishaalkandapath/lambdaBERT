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

import threading

class ThreadLockDict:
    def __init__(self):
        self.lock = threading.Lock()
        self.dict = {}
    
    def __getitem__(self, key):
        with self.lock:
            return self.dict[key]
    
    def __setitem__(self, key, value):
        with self.lock:
            self.dict[key] = value
    
    def __delitem__(self, key):
        with self.lock:
            del self.dict[key]
    
    def __contains__(self, key):
        with self.lock:
            return key in self.dict
    
    def __len__(self):
        with self.lock:
            return len(self.dict)
    
    def __iter__(self):
        with self.lock:
            return iter(self.dict)

#global variables
thread_locked_dict = ThreadLockDict()

#model inference
def model_inference(model, dataloader):
    global DEVICE
    model.eval()
    confusion_matrix = torch.zeros(5, 5)
    average_loss = 0
    count = 0
    outs = []
    cls_outs = []

    prs, ps = [], []
    word_prs, var_prs, lambda_prs, app_prs = [], [], [], []
    word_ps, var_ps, lambda_ps, app_ps = [], [], [], []
    with torch.no_grad():
        #prpgress bar
        pbar = tqdm.tqdm(total=len(dataloader))
        for k, batch in enumerate(dataloader):
            (in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, stop_mask, pad_mask, sent_pad_mask) = batch
            #move to device
            in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, stop_mask, pad_mask, sent_pad_mask = in_embs.to(DEVICE), target_embs.to(DEVICE), target_tokens.to(DEVICE), var_index_mask_no.to(DEVICE), lambda_index_mask.to(DEVICE), app_index_mask.to(DEVICE), stop_mask.to(DEVICE), pad_mask.to(DEVICE), sent_pad_mask.to(DEVICE)
            
            seq_syntax = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) 
            out, classified_class, var_reg = model(target_embs[:, :-1, :], seq_syntax[:, :-1], in_embs, sent_pad_mask) # get_discrete_output(in_embs, model, target_tokens.shape[1])
            target = target_embs[:, 1:, :]
            lambda_index_mask, app_index_mask, var_index_mask_no, stop_mask, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], stop_mask[:, 1:], pad_mask[:, 1:])
        
            #classiifer truth
            gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) #+ 4*stop_mask.type(torch.bool) #because lambda's class is 2
            loss = nn.functional.cross_entropy(classified_class.view(-1, 4), gt_cls_mask.view(-1), reduction="none")
            loss = ((loss) * (0.95 ** (torch.arange(loss.shape[0])).to(gt_cls_mask.device))).mean()# -- discounted loss
            average_loss += loss.item()
            count += 1
            pbar.set_description(f"Loss: {loss.item()/count}")
            pbar.update(1)
            #add to confusion matrix
            classified_class_ = classified_class.argmax(dim=-1)
            for i in range(4):
                for j in range(4):
                    confusion_matrix[i, j] += ((classified_class_ == j) & (gt_cls_mask == i)).sum().detach().cpu()

            #probability of true sequence:
            # pr = torch.nn.Softmax(dim=-1)(classified_class).max(dim=-1)[0].prod(dim=-1).squeeze(0).item()
            pr = torch.nn.Softmax(dim=-1)(classified_class)
            pr = torch.gather(pr, -1, gt_cls_mask.unsqueeze(-1)).squeeze(-1).prod(dim=-1).squeeze(0).item()

            #Write the written outputs:
            out = out[0].argmax(-1) if len(out.shape) == 3 else out[0]
            outs.append([classified_class_.squeeze(0).tolist(), get_out_list(out, classified_class, var_reg), pr])
            
            beam_size = 12
            x, y, z, p = get_discrete_output(in_embs, model, tokenized=True, last=False, max_len=200, unfiltered_class=True, beam_size=beam_size)
            if beam_size <= 1: outs.append([y.argmax(-1).squeeze(0).tolist(), get_out_list(x, y, z), p])
            else:
                for jkl in range(len(y)):
                    outs.append([y[jkl].tolist(), get_out_list(x[jkl], y[jkl], z[jkl]), p[jkl].item()])
                outs.append(["", "", ""]) # emmpty divider

            if not beam_size: 
                classified_class_ = torch.nn.Softmax(dim=-1)(classified_class).cpu()
                word_prs.append(classified_class_[0, :, 0])
                var_prs.append(classified_class_[0, :, 1])
                lambda_prs.append(classified_class_[0, :, 2])
                app_prs.append(classified_class_[0, :, 3])

                y = torch.nn.Softmax(dim=-1)(y).cpu()
                word_ps.append(y[0, :, 0])
                var_ps.append(y[0, :, 1])
                lambda_ps.append(y[0, :, 2])
                app_ps.append(y[0, :, 3])

            prs.append(pr)
            ps.append(p) if beam_size <= 1 else ps.extend(p)
            if k>10: break
    # #write
    loss = average_loss / count
    print("Average Loss: ", loss)

    csv_file = open("outputs_gahhhh.csv", "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(outs)
    csv_file.close()

    plot_teacher_forcing_error(prs, ps, save_as="teaching_forcing_not_last.png")
    # mean_probability_measures(word_prs, word_ps, title="Evolution of Word Probabilities", save_as="word_time_notlast.png")
    # mean_probability_measures(var_prs, var_ps, title="Evolution of Var Probabilities", save_as="var_time_notlast.png")
    # mean_probability_measures(lambda_prs, lambda_ps, title="Evolution of Lambda Probabilities", save_as="lmda_time_notlast.png")
    # mean_probability_measures(app_prs, app_ps, title="Evolution of App Probabilities", save_as="app_time_notlast.png")


    return confusion_matrix

def plot_teacher_forcing_error(true_prs, inference_prs, save_as=""):
    diff = np.array(true_prs) - np.array(inference_prs)

    # Create histogram using seaborn for better default styling
    plt.figure(figsize=(10, 6))
    sns.histplot(diff, bins=30, kde=True)
    
    # Add labels and title
    plt.xlabel('Probability Difference (True - Inference)')
    plt.ylabel('Count')
    plt.title('Distribution of Teaching Forcing vs Inference Probability Differences')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_as)

def mean_probability_measures(true_probs, inference_probs, title="Evolution of Var Probabilities", save_as=""):
    """
    Create a bar plot comparing mean probabilities at each position, excluding -1 values.
    Args:
        true_probs: Matrix of shape B x N with true probabilities
        inference_probs: Matrix of shape B x N with inference probabilities
        title: Plot title
    """

    max_len = max(max(len(seq) for seq in true_probs), 
                  max(len(seq) for seq in inference_probs))
    
    def pad_sequence(seq, max_len):
        return np.pad(seq, (0, max_len - len(seq)), 
                     mode='constant', 
                     constant_values=-1)
    
    # Convert lists to padded numpy arrays
    true_probs = np.array([pad_sequence(seq, max_len) for seq in true_probs])
    inference_probs = np.array([pad_sequence(seq, max_len) for seq in inference_probs])
    
    # Get number of positions (N)
    n_positions = true_probs.shape[1]
    
    # Calculate means at each position, excluding -1s
    true_means = []
    inference_means = []
    
    for pos in range(n_positions):
        # Get values at current position
        true_pos_vals = true_probs[:, pos]
        inf_pos_vals = inference_probs[:, pos]
        
        # Filter out -1 values
        true_valid = true_pos_vals[true_pos_vals != -1]
        inf_valid = inf_pos_vals[inf_pos_vals != -1]
        
        # Calculate means (if there are valid values)
        true_mean = np.mean(true_valid) if len(true_valid) > 0 else 0
        inf_mean = np.mean(inf_valid) if len(inf_valid) > 0 else 0
        
        true_means.append(true_mean)
        inference_means.append(inf_mean)
    
    # Convert to numpy arrays
    true_means = np.array(true_means)
    inference_means = np.array(inference_means)
    
    # Create positions for bars
    x = np.arange(n_positions)
    width = 0.35  # Width of bars
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, true_means, width, label='True Probabilities', alpha=0.7)
    plt.bar(x + width/2, inference_means, width, label='Inference Probabilities', alpha=0.7)
    
    # Customize plot
    plt.xlabel('Position')
    plt.ylabel('Mean Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    plt.xticks(x)
    
    plt.savefig(save_as)

# plot confusion matrix
def plot_confusion_matrix(confusion_matrix, split="train"):
    #indx to label map:
    #normalize confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)
    #make a color bar
    label_map = {0: "Word", 1: "Variable", 2: "Lambda", 3: "Application"}#, 4: "Stop"}
    plt.imshow(confusion_matrix, cmap="viridis", )
    plt.xticks(list(label_map.keys()), list(label_map.values()))
    plt.yticks(list(label_map.keys()), list(label_map.values()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(f"confusion_matrix_{split}.png")
    plt.clf()

def easy_free_variable_counts(input_sents, model):
    model.eval()

    tokenized = TOKENIZER(input_sents, return_tensors="pt", padding=True)
    input_embs = get_bert_emb(tokenized)
    input_embs = input_embs.to(DEVICE)

    out, classified_class, var_reg = model(input_embs)
    classified_class = classified_class.argmax(dim=-1).squeeze(0)
    lambda_indices, app_indices, var_indices  = torch.where(classified_class == 2), torch.where(classified_class == 3), torch.where(classified_class == 1)

    # for every vector at a variable position, check if it is similar to atleast one variable next to a lambda location
    #pairwise similarity of tokens of each batch
    sim = torch.nn.CosineSimilarity()(var_reg)

    var_mask = (classified_class == 1) @ (classified_class == 1).T
    bound_var_pos = torch.roll(classified_class == 2, 1, 1).repeat(1, 1, classified_class.shape[1])

    sim = (sim*var_mask*bound_var_pos) >= 0.85
    sim = sim.sum(dim=1) != 0

    unbound_count = torch.count_nonzero(sim[var_indices] == 0)

    return unbound_count

def scope_violation_count(words):
    open_brackets = 0
    unclosed_variables = {}

    remove_from_scope = lambda x: [unclosed_variables.pop(key) for key in unclosed_variables if unclosed_variables[key] == x]

    out_of_bounds = 0
    prev = None
    for w in words:
        if w == "(": 
            open_brackets += 1
            prev=None
        elif "@@VAR@@" in w and prev == "λ": 
            unclosed_variables[w] = open_brackets
        elif ("@@VAR@@" in w and w not in unclosed_variables) or (w != "λ" and prev != "λ"): 
            out_of_bounds+=1
            if prev: 
                prev = None
                open_brackets -= 1
                remove_from_scope(open_brackets)
            else: prev=w
        elif w == "λ": prev = w

    return out_of_bounds


def get_discrete_output(input_embs, model, max_len=20, tokenized=False, last=False, unfiltered_class=False, beam_size=1):
    model.eval()
    #bert tokenize and get embeddings
    if not tokenized: 
        tokenized = TOKENIZER(input_embs, return_tensors="pt", padding=True)
        input_embs, _ = get_bert_emb(tokenized)
    input_embs = input_embs.to(DEVICE)
    #model inference
    out, classified_class, var_reg, pr = model(input_embs, max_len=max_len, last=last, unfiltered_class=unfiltered_class, beam_size=beam_size)
    return out, classified_class, var_reg, pr

def get_out_list(out, classified_class, var_reg):
    classified_class = classified_class.argmax(dim=-1).squeeze(0)
    var_reg = var_reg.squeeze(0)
    out_list = TOKENIZER.convert_ids_to_tokens(out) 
    lambda_indices, app_indices, var_indices  = torch.where(classified_class == 2)[0].tolist(), torch.where(classified_class == 3)[0].tolist(), torch.where(classified_class == 1)[0].tolist()

    for i in lambda_indices:
        out_list[i] = "λ"
    for i in app_indices:
        out_list[i] = "("
    
    # time for variables
    var_dict = {}
    for i in var_indices:
        if len(var_dict) == 0:
            var_dict[0] = var_reg[i]
            out_list[i] = f"x{len(var_dict)}"
        else:
            min_sim = 0
            min_key = 0
            for key in var_dict:
                sim = torch.nn.functional.cosine_similarity(var_reg[i], var_dict[key], dim=0)
                if sim > min_sim:
                    min_sim = sim
                    min_key = key
            if min_sim > 0.9:
                out_list[i] = f"x{min_key}"
            else:
                var_dict[len(var_dict)] = var_reg[i]
                out_list[i] = f"x{len(var_dict)}"
    return " ".join(out_list)


def levenstein_lambda_term(str1, str2):
    """
    Basically the same as normal levenstein distance, but with the added nuance of comparing variables naming knowing that they can be different and that's ok
    Variables are to be treated as pointers to all instances of that variable. Further, Variable counts should match.

    Essential details:
     - analyze in components as a whole not per character: hence in is a list of individual tokens
     - if comparing a variable against a variable and their cardinalities match: do one that replaces and one that does not
     - the above comparison increases the number of options at a point conditionally, bringin the dp formula to :
        lev(a, b) = 1. |a| if |b| = 0
                    2. |b| if |a| = 0
                    3. lev(tail(a), tail(b)) if head(a) == head(b)
                    4. 1 + min(lev(tail(a), b), lev(a, tail(b)), lev(tail(a), tail(b))) if type(a) != var and type(b) != var and not car(a) == car(b)
    - changing variable declaration spots with each other does not lead to a penalty. Changes in variable usages do - coz a change in variale declaration naming should be reflected in all usages because pointers
    """
    num_vars_a = sum([1 for t in str1 if re.match(r"(S|NP|N|PP)_\d+", t)])
    num_vars_b = sum([1 for t in str2 if re.match(r"(S|NP|N|PP)_\d+", t)])
    rename = lambda x, t: x[:re.findall(r"(S|NP|N|PP)_\d+", x)[0].find("_")+1]+ (int(x[re.findall(r"(S|NP|N|PP)_\d+", x)[0].find("_")+1:]) + t)
    new_str2 = [t if not re.match(r"(S|NP|N|PP)_\d+", t) else rename(t, num_vars_a) for t in str2] # offset so that no two variables are shared acorss str1 and str2

    var_name_counter = num_vars_a + num_vars_b

    str2_var_pointers = {}
    new_new_str2 = []
    for i, t in enumerate(new_str2):
        if re.match(r"(S|NP|N|PP)_\d+", t):
            if t in str2_var_pointers: 
                new_new_str2.append(str2_var_pointers[t])
            else:
                str2_var_pointers[t] = [t]
                new_new_str2.append(str2_var_pointers[t])
    new_str2 = tuple(new_new_str2)

    #initialize dp 
    distances = np.zeros((len(str1) + 1, len(new_new_str2) + 1))

    for t1 in range(len(str1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(new_new_str2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0

    access_b = lambda i: new_new_str2[i][0] if isinstance(new_new_str2[i], list) else new_new_str2[i]
    
    for t1 in range(1, len(str1) + 1):
        for t2 in range(1, len(new_new_str2) + 1):
            if (str1[t1-1] == access_b(t2-1)):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            elif t1 >= 2  and t2 >= 2 \
                and str1[t1-2] == "λ" and access_b(t2-2) == "λ" \
                and (re.match(r"(S|NP|N|PP)_\d+", str1[t1-1]) and re.match(r"(S|NP|N|PP)_\d+", access_b(t2-1))):
                c = distances[t1 - 1][t2 - 1] # no need to change anything here
                #switch pointers around in b 
                #make new var name
                new_var_name = f"NP_{var_name_counter}"
                var_name_counter += 1
                str2_var_pointers[access_b(t2-1)][0] = var_name_counter
                str2_var_pointers[new_var_name] = str2_var_pointers[access_b(t2-1)]
                str1.replace(str1[t1-1], new_var_name)

                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(str1)][len(new_new_str2)]



class ClassifierModel(nn.Module):
    def __init__(self, model, linear):
        super().__init__()
        self.model = model
        self.linear = linear
    
    @dispatch(torch.Tensor)
    def forward(self, in_embs, max_len=20, classify=True, beam_size=0, last=False, unfiltered_class=False):
        # NOTE: classify shud be deprecated
        if beam_size >= 1: 
            return self.beam_search_inference(in_embs, max_len=max_len, classify=classify, beam_size=beam_size)
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
                out_stacked = torch.cat([out_stacked, out[:, -1].unsqueeze(1)], dim=1)
                classified_class_stacked = torch.cat([classified_class_stacked, classified_class_[:, -1].unsqueeze(1)], dim=1)
            else:
                out_stacked = torch.cat([out_stacked, out[:, -1].unsqueeze(1)], dim=1)
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

    def beam_search_inference(self, in_embs, max_len=20, classify=True, beam_size=1): # TODO: make beam start from 5 -- number of classes
        x = torch.tensor(BOS_TOKEN_LAST).to(in_embs.device)
        out_stacked, classified_class_stacked, var_reg_stacked, newest_out = x, torch.tensor([[0]]).long().to(x.device) if classify else None, None, None
        list_out = []
        cls_out = []
        var_out = []
        prob_list = []
        while out_stacked.size(1) < max_len:
            in_embs = torch.repeat_interleave(in_embs[0:1], x.size(0), dim=0).to(x.device)
            # process each beam 
            out, classified_class, var_reg = self.model(x, classified_class_stacked, in_embs, mb_pad=torch.zeros(in_embs.shape[:-1]).to(x.device).to(torch.bool), device=x.device)
            if classify: 
                #sort for beams
                classified_class = nn.functional.softmax(classified_class, dim=-1)
                cls_probs, classified_class = classified_class.sort(-1, descending=True) # batch x length x 4
                classified_class = classified_class[:, :, :beam_size] # batch x length x beam
                cls_probs = cls_probs[:, :, :beam_size] # batch x length x beam
                
                #make beams
                if prob_list == []: #first time
                    classified_class = classified_class.squeeze(0).T # beam x length -- because batch is 1
                    cls_probs = cls_probs.squeeze(0).T # similarly 
                    prob_list.extend(sum(cls_probs.tolist(), start=[])) #get the different beam probabilities -- not gonna log coz it might be to small or smth
                else:
                    # get the beam best from each, flatten, take beam best 
                    #beam x length x 1 * batch x beam x length
                    #prod the prbabilities to get the sequences probability so far
                    cls_probs, classified_class_ = (torch.tensor(prob_list).to(x.device).unsqueeze(-1) * cls_probs.transpose(-1, -2)[:, : , -1]).flatten().sort(descending=True)
                    #at this point i have sorted for each of the previous beams, its best succeeding beams. 
                    classified_class_ = classified_class_[:beam_size]
                    b_indices = classified_class_//min(beam_size, classified_class.size(-1)) #which previous beam is it from?
                    cl_indices = classified_class_ % min(beam_size, classified_class.size(-1)) #which new beam is it?

                    cls_probs = cls_probs[:beam_size]

                    #classified class needs to be beam x length
                    classified_class = classified_class[b_indices, :, cl_indices]

                    #modify prob_list
                    prob_list = torch.tensor(prob_list).to(b_indices.device)[b_indices] if type(prob_list) is list else prob_list[b_indices]
                    prob_list *= cls_probs
                    # print(prob_list, cls_probs)

            if var_reg_stacked is None:
                var_reg_stacked = var_reg
                out_stacked = torch.cat([out_stacked, out[:, -1].unsqueeze(1)], dim=1) 
                out_stacked = out_stacked.squeeze(0).repeat(min(beam_size, 4), 1, 1)# because the first remains for all the beams
                var_reg_stacked = var_reg_stacked.squeeze(0).repeat(min(beam_size, 4), 1, 1)
                classified_class_stacked = classified_class_stacked.squeeze(0).repeat(min(beam_size, 4), 1)
                classified_class_stacked = torch.cat([classified_class_stacked, classified_class[:, -1].unsqueeze(1)], dim=1)
                
            else:
                out_stacked = torch.cat([out_stacked[b_indices], out[b_indices, -1].unsqueeze(1)], dim=1) # get the outputs of these batches only
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
                    # print(TOKENIZER.convert_ids_to_tokens([newest_out]))
                    if newest_out == 102: # and it is SEP
                        if (len(list_out) == beam_size and list_out[0][0] < prob_list[i]) or (len(list_out) < beam_size): 
                            if len(list_out) == beam_size: list_out = list_out[1:]
                            list_out.append((prob_list[i], random.random(), self.get_words(out_stacked[i], classified_class_stacked[i]), 
                                                                        classified_class_stacked[i, 1:],
                                                                        var_reg_stacked[i])) # save the sequence
                            list_out.sort() # maintain ordering
                            rem_list.append(i)
                            continue
                not_rem.append(i)
                assert len(list_out) <= beam_size, len(list_out)
            if len(rem_list) == beam_size: break

            for i in rem_list:
                prob_list[i] = prob_list[not_rem[0]]
                out_stacked[i] = out_stacked[not_rem[0]]
                var_reg_stacked[i] = var_reg[not_rem[0]]
                classified_class_stacked[i] = classified_class_stacked[not_rem[0]]

            x = out_stacked
            assert len(list_out) <= beam_size, len(list_out)
        
        if out_stacked.size(1) == max_len and len(list_out) != beam_size:
            #gotta do one last check for the best sequeces:
            for i in range(out_stacked.size(0)):
                # if (classified_class_stacked[i, -1] == 0): # if we have a word
                #     sorted_stiff, newest_out = self.linear(out_stacked[i, -1, :]).sort(-1)
                #     newest_out = newest_out[-1].item()
                if len(list_out) < beam_size or prob_list[i] > list_out[0][0]:
                    if len(list_out) == beam_size: list_out = list_out[1:]# replace!
                    list_out.append((prob_list[i], random.random(), self.get_words(out_stacked[i], classified_class_stacked[i]),
                    classified_class_stacked[i, 1:], 
                    var_reg_stacked[i]
                    ))
                    list_out.sort(key=lambda x: x[0]) # maintain ordering
        assert len(list_out) <= beam_size, len(list_out)
        list_out.sort(key=lambda x: x[0], reverse=True)
        ps, _, list_out, cls_out, var_out = (zip(*list_out))   
        return list_out, cls_out, var_out, ps

    @dispatch(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)   
    def forward(self, seq, seq_syntax, in_embs, mb_pad):
        outs, classified_class, var_emb = self.model(seq, seq_syntax, in_embs, mb_pad=mb_pad , device=seq.device)
        return self.linear(outs), classified_class, var_emb

def test_data_methods(dataset):
    item = dataset[0]
    (in_embs, target_embs, target_tokens, lambda_index_mask, var_index_mask_no, app_index_mask) = item

    #try and decode target otkens back 
    target_tokens = [101 if target_tokens[i] == -1 else target_tokens[i] for i in range(len(target_tokens))]

    target_decoded = TOKENIZER.convert_ids_to_tokens(target_tokens)
    print(" ".join(target_decoded))

def parallelize_inference(i, model, words, max_len=200, last=False):
    global thread_locked_dict
    x = get_out_list(*get_discrete_output(words, model, max_len=200, last=False))
    thread_locked_dict[i] = x


if __name__ == "__main__":
    import os
    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--last", action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    #load model
    model = TransformerDecoderStack(4, 384, 8, 3072)
    checkpoint = torch.load(args.model_path)
    model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model_weights.update({k: torch.zeros_like(v) for k, v in model.state_dict().items() if k not in model_weights})
    model.load_state_dict(model_weights)

    if any([not k.startswith("model.") for k in checkpoint["state_dict"].keys()]):
        linear = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.LayerNorm(768, 1e-12), nn.Linear(768, TOKENIZER.vocab_size))
        linear_weights = {k.replace("linear.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("linear.")}
        linear.load_state_dict(linear_weights)
        
        model = ClassifierModel(model, linear)
    else: model = ClassifierModel(model, BertForMaskedLM(BERT_MODEL.config).cls)

    # model = ShuffledTransformerStack.load_from_checkpoint(args.model_path, model).model
    DEVICE = torch.device("cpu") if args.cpu else torch.device("cuda")
    model = model.to(DEVICE)

    # --LOAD DATA
    dataloader, valid_dataloader, test_dataloader = dataloader.data_init(1, last=args.last)


    #-- CONFUSION MATRIX --
    # confusion_matrix = model_inference(model, test_dataloader)
    # plot_confusion_matrix(confusion_matrix, "test")

    # confusion_matrix = model_inference(model, valid_dataloader)
    # plot_confusion_matrix(confusion_matrix, "valid")

    confusion_matrix = model_inference(model, dataloader)
    plot_confusion_matrix(confusion_matrix)

    # --DISCRETE OUTPUT SAMPLES
    # lines = pd.read_csv("data/input_sentences.csv", header=None)
    # out_file = open("data/output_samples.csv", "a")
    # out_file_csv = csv.writer(out_file)

    # # rand_lines = random.choices(range(len(lines)))
    # write_lines = []
    # for r_line in tqdm.tqdm(range(0, len(lines), 15)):
    #     w_words = []
    #     for rand_line in range(r_line, min(r_line+15, len(lines))):
    #         line = eval(lines.iloc[rand_line, 1])
    #         target_path = lines.iloc[rand_line, 2][11:]
    #         target_text = open(target_path, "r").readlines()[0]
    #         words = " ".join(line).replace("...}"," ...}").replace("{..","{. .").replace("NP.","NP .").replace("NP—","NP —").replace(",}"," ,}").\
    #             replace("'re"," 're").replace("'s"," 's").replace("'ve}"," 've}").replace("!}"," !}").replace("?}"," ?}").replace("n't"," n't").\
    #             replace("'m}"," 'm}").replace("{. ..","{...").replace("{——}","{— —}").replace("{--—}","{- -—}").replace("St.", "St").split()
    #         words = [word[:-1] for i, word in enumerate(words) if i % 2 != 0]
    #         words = " ".join(words)
        
    #         w_words.append(words)

    #     #parallelize
    #     threads = []
    #     for i, words in enumerate(w_words):
    #         t = threading.Thread(target=parallelize_inference, args=(i, model, words, 200, args.last))
    #         threads.append(t)
    #         t.start()
        
    #     for t in threads:
    #         t.join()
        
    #     for i in range(len(w_words)):
    #         write_lines.append([w_words[i], thread_locked_dict[i]])
        
    #     #clear thread_locked dict:
    #     w_words = []
    #     thread_locked_dict = ThreadLockDict()

    #     if r_line % 90 == 0:
    #         out_file_csv.writerows(write_lines)
    #         out_file.flush()
    #         os.fsync(out_file)
    #         write_lines = []    




