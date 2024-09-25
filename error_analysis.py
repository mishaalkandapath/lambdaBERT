import matplotlib.pyplot as plt
from models import *
import dataloader
import tqdm, csv, random, pandas as pd
from tokenization import get_bert_emb, TOKENIZER, LAMBDA, OPEN_RRB, BERT_MODEL
from dataloader import SEP_TOKEN, BOS_TOKEN

import torch 
import torch.nn as nn
import numpy as np

import re, copy
from transformers import BertForMaskedLM, BertConfig
from multipledispatch import dispatch

SEP_ID=102
BOS_ID=101

#model inference
def model_inference(model, dataloader):
    global DEVICE
    model.eval()
    confusion_matrix = torch.zeros(4, 4)
    with torch.no_grad():
        #prpgress bar
        for batch in tqdm.tqdm(dataloader):
            (in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask, _) = batch
            #move to device
            in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask = in_embs.to(DEVICE), target_embs.to(DEVICE), target_tokens.to(DEVICE), var_index_mask_no.to(DEVICE), lambda_index_mask.to(DEVICE), app_index_mask.to(DEVICE), pad_mask.to(DEVICE)

            out, classified_class, var_reg = model(target_embs[:, :-1, :], in_embs)
            target = target_embs[:, 1:, :]
            lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

            #classiifer truth
            gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) #because lambda's class is 2

            #add to confusion matrix
            classified_class = classified_class.argmax(dim=-1)
            for i in range(4):
                for j in range(4):
                    confusion_matrix[i, j] += ((classified_class == j) & (gt_cls_mask == i)).sum().detach().cpu()
    return confusion_matrix

# plot confusion matrix
def plot_confusion_matrix(confusion_matrix):
    #indx to label map:
    #normalize confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)
    #make a color bar
    label_map = {0: "Word", 1: "Variable", 2: "Lambda", 3: "Application"}
    plt.imshow(confusion_matrix, cmap="viridis", )
    plt.xticks(list(label_map.keys()), list(label_map.values()))
    plt.yticks(list(label_map.keys()), list(label_map.values()))
    plt.colorbar()
    plt.savefig("confusion_matrix.png")

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


def get_discrete_output(input_sent, model):
    model.eval()
    #bert tokenize and get embeddings
    tokenized = TOKENIZER(input_sent, return_tensors="pt", padding=True)
    input_embs, _ = get_bert_emb(tokenized)
    input_embs = input_embs.to(DEVICE)
    #model inference
    out, classified_class, var_reg = model(input_embs)
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
    def forward(self, in_embs):
        x = torch.tensor(BOS_TOKEN).to(in_embs.device)
        out_stacked, classified_class_stacked, var_reg_stacked, newest_out = None, None, None, None
        list_out = []
        while newest_out != 102 and len(list_out) < 20:
            out, classified_class, var_reg = self.model(x, in_embs, mb_pad=torch.zeros_like(in_embs), device=x.device)
            if out_stacked is None:
                out_stacked, classified_class_stacked, var_reg_stacked = out, classified_class, var_reg
            else:
                out_stacked = torch.cat([out_stacked, out[:, -1].unsqueeze(1)], dim=1)
                classified_class_stacked = torch.cat([classified_class_stacked, classified_class[:, -1].unsqueeze(1)], dim=1)
                var_reg_stacked = torch.cat([var_reg_stacked, var_reg[:, -1].unsqueeze(1)], dim=1)
            

            cls_probs = nn.functional.softmax(classified_class[-1, -1], dim=-1)
            # print(cls_probs.sort(-1))
            if classified_class[-1, -1].argmax() == 0:
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
        return list_out, classified_class_stacked, var_reg_stacked
    
    @dispatch(torch.Tensor, torch.Tensor)   
    def forward(self, seq, in_embs):
        outs, classified_class, var_emb = self.model(seq, in_embs, mb_pad=torch.zeros_like(in_embs), device=seq.device)
        return self.linear(outs), classified_class, var_emb

# def random_test(model, dataloader):
#     model = model.model
#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm.tqdm(dataloader):
#             (in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask) = batch
#             #move to device
#             in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask = in_embs.to(DEVICE), target_embs.to(DEVICE), target_tokens.to(DEVICE), var_index_mask_no.to(DEVICE), lambda_index_mask.to(DEVICE), app_index_mask.to(DEVICE), pad_mask.to(DEVICE)

#             out, classified_class, var_reg = model(target_embs[:, :-1, :], in_embs)
#             target = target_embs[:, 1:, :]
#             lambda_index_mask, app_index_mask, var_index_mask_no, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], pad_mask[:, 1:])

#             #classiifer truth
#             gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool)
#             break

def test_data_methods(dataset):
    item = dataset[0]
    (in_embs, target_embs, target_tokens, lambda_index_mask, var_index_mask_no, app_index_mask) = item

    #try and decode target otkens back 
    target_tokens = [101 if target_tokens[i] == -1 else target_tokens[i] for i in range(len(target_tokens))]

    target_decoded = TOKENIZER.convert_ids_to_tokens(target_tokens)
    print(" ".join(target_decoded))



if __name__ == "__main__":
    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    #load model
    model = TransformerDecoderStack(4, 384, 8, 3072)
    checkpoint = torch.load(args.model_path)
    model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
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
    dataloader = dataloader.data_init(70, mode=3)


    #-- CONFUSION MATRIX --
    confusion_matrix = model_inference(model, dataloader)
    plot_confusion_matrix(confusion_matrix)

    #--DISCRETE OUTPUT SAMPLES
    # lines = pd.read_csv("data/input_sentences.csv", header=None)
    # out_file = open("data/output_samples.csv", "w")
    # out_file_csv = csv.writer(out_file)

    # rand_lines = random.choices(range(len(lines)), k=10)
    # write_lines = []
    # for rand_line in rand_lines:
    #     line = eval(lines.iloc[rand_line, 1])
    #     target_path = lines.iloc[rand_line, 2][11:]
    #     target_text = open(target_path, "r").readlines()[0]
    #     words = " ".join(line).replace("...}"," ...}").replace("{..","{. .").replace("NP.","NP .").replace("NP—","NP —").replace(",}"," ,}").\
    # replace("'re"," 're").replace("'s"," 's").replace("'ve}"," 've}").replace("!}"," !}").replace("?}"," ?}").replace("n't"," n't").\
    # replace("'m}"," 'm}").replace("{. ..","{...").replace("{——}","{— —}").replace("{--—}","{- -—}").replace("St.", "St").split()
        
    #     words = [word[:-1] for i, word in enumerate(words) if i % 2 != 0]
    #     words = " ".join(words)
    #     print("Target Text: ", target_text)
    #     out = get_discrete_output(words, model)
    #     print("------\n")

    #     # print(words)
        
    #     # print(out)
    #     write_lines.append([words, out])
    # out_file_csv.writerows(write_lines)




