import os
import waitGPU
if __name__ == "__main__":
    waitGPU.wait(memory_ratio=0.001,
                gpu_ids=list(range(2)), interval=10, nproc=1, ngpu=1)
import pickle
import matplotlib.pyplot as plt

from models import *
import dataloader
import tqdm, csv, pandas as pd
from tokenization import TOKENIZER
from dataloader import (BOS_TOKEN, BOS_TOKEN_LAST, BOS_ID, SEP_ID)
from inference_model import InferenceModel, get_closest_idx, get_closest_var_idx

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

import time
from utils import ThreadLockDict

DEVICE = torch.device("cuda:0")

SPEC_TOKENS = {"multilingual_bert": {"lambda": 475, "app": 113},
                "bert_base": {"lambda": 1165, "app": 1006},
                "roberta_base": {"lambda": 48191, "app":1640}}

def clean_lambda_tokens(text):
    # Add spaces around lambda calculus symbols
    text = re.sub(r'([()λ])', r' \1 ', text)  # Space around parens and lambda
    text = re.sub(r'\s+', ' ', text)  # Clean up multiple spaces
    text = text.strip()
    return text

def get_mapped_out_sequences(in_tokens, out_tokens, spec_tokens=False):
    in_sentence = TOKENIZER[os.environ["BERT_TYPE"]].convert_ids_to_tokens(in_tokens)
    out_sentence = TOKENIZER[os.environ["BERT_TYPE"]].convert_ids_to_tokens(out_tokens)
    assert in_tokens == TOKENIZER[os.environ["BERT_TYPE"]].encode(" ".join(in_sentence)), f"{in_tokens} != {TOKENIZER[os.environ['BERT_TYPE']].encode(' '.join(in_sentence), add_special_tokens=False)}"
    if spec_tokens:
        in_sentence = in_sentence[1:-1]
        out_sentence = out_sentence[1:-1]

    in_sentence_mapping = TOKENIZER[os.environ["BERT_TYPE"]](" ".join(in_sentence), return_offsets_mapping=True)
    out_sentence_mapping = TOKENIZER[os.environ["BERT_TYPE"]](" ".join(out_sentence), return_offsets_mapping=True)
    pass

def obtain_target_lambda_sequence(
    in_tokens, var_indices_no, 
    lambda_indices, app_indices): # 1165, 1006 
    in_tokens[lambda_indices] = SPEC_TOKENS[os.environ["BERT_TYPE"]]["lambda"]
    in_tokens[app_indices] = SPEC_TOKENS[os.environ["BERT_TYPE"]]["app"]
    in_tokens[in_tokens == -1] = 0

    #mark the var_indices
    var_indices = torch.where(var_indices_no != 0)[0]
    in_tokens = in_tokens.tolist()
    offset = 0
    for i in var_indices.tolist():
        t = TOKENIZER[os.environ["BERT_TYPE"]].encode(f"x{var_indices_no[i]}" if "roberta" not in os.environ["BERT_TYPE"] else f" x{var_indices_no[i]} ", add_special_tokens=False)
        in_tokens = in_tokens[:i+offset] + t + in_tokens[i+1+offset:]
        offset += len(t) - 1
    in_tokens = torch.tensor(in_tokens)

    word_list = TOKENIZER[os.environ["BERT_TYPE"]].decode(in_tokens)
    word_list = clean_lambda_tokens(word_list)
    word_list = word_list.split()
    return word_list[1:-1]

def get_out_list(out, classified_class, var_reg,
                  in_embs, in_tokens, dynamic_vars=False,
                    get_indices=False,
                    ret_token_indices=False):
    out_list = []
    out_indices = []
    if len(classified_class.shape) > 1: classified_class = classified_class.argmax(dim=-1).squeeze(0)
    for i in range(out.size(0)):
        leftover_words = torch.count_nonzero(classified_class[:i] == 0) == in_embs.size(0) - 2
        if classified_class[i] == 0:
            cl_idx = get_closest_idx(out[i, :], in_embs, in_tokens, )
            token_idx = in_tokens[cl_idx]
            out_list.append(token_idx.item())
            out_indices.append(cl_idx)
        else:
            out_list.append(BOS_ID[os.environ["BERT_TYPE"]])
            out_indices.append(-1)
    assert len(out_list) == out.size(0)
    assert out.size(0) == classified_class.size(0), f"{out.shape} {classified_class.shape}"

    if dynamic_vars: var_reg = var_reg.squeeze(0)
    
    lambda_indices, app_indices, var_indices = torch.where(classified_class == 2)[0].tolist(), torch.where(classified_class == 3)[0].tolist(), torch.where(classified_class == 1)[0].tolist()

    for i in lambda_indices:
        out_list[i] = SPEC_TOKENS[os.environ["BERT_TYPE"]]["lambda"]#"λ"

    for i in app_indices:
        out_list[i] = SPEC_TOKENS[os.environ["BERT_TYPE"]]["app"]#"("
    
    # time for variables
    offset = 0
    if dynamic_vars:
        var_dict = {}
        for i in var_indices:
            if len(var_dict) == 0 or out_list[i-1] == "λ": 
                out_list[i] = f"x{len(var_dict)}"
                var_dict[len(var_dict)] = var_reg[i]
            else:
                all_vars = list(var_dict.values())
                #choose the variable w the highest similarity to the current one:
                sim = torch.nn.functional.cosine_similarity(var_reg[i].unsqueeze(0), torch.stack(all_vars), dim=1)
                max_sim_index = sim.argmax().item()
                out_list[i] = f"specvar{max_sim_index}"
    else:
        for i in var_indices:
            out_emb = out[i]
            var_idx = get_closest_var_idx(out_emb, var_count=i)
            t = TOKENIZER[os.environ["BERT_TYPE"]].encode(f" specvar{var_idx} ", add_special_tokens=False)
            out_list = out_list[:i+offset] + t + out_list[i+1+offset:]
            offset += len(t) - 1
            # out_list[i] = f"x{var_idx}"

    if not ret_token_indices:
        out_list = TOKENIZER[os.environ["BERT_TYPE"]].decode(out_list) 
        if "roberta" in os.environ["BERT_TYPE"]: out_list = clean_lambda_tokens(out_list)
        out_list = out_list.split()
        if get_indices:
            return " ".join(out_list), out_list
        return " ".join(out_list)
    else:
        out_list = [TOKENIZER[os.environ["BERT_TYPE"]].decode(st) for st in out_list]
        return out_list, out_indices
        

def teacher_forcing(model, batch):
    global BOS_TOKEN_LAST, BOS_TOKEN
    (in_tokens, in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, stop_mask, pad_mask, sent_pad_mask) = batch
    in_tokens, in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, stop_mask, pad_mask, sent_pad_mask = in_tokens.to(DEVICE), in_embs.to(DEVICE), target_embs.to(DEVICE), target_tokens.to(DEVICE), var_index_mask_no.to(DEVICE), lambda_index_mask.to(DEVICE), app_index_mask.to(DEVICE), stop_mask.to(DEVICE), pad_mask.to(DEVICE), sent_pad_mask.to(DEVICE)

    true_tokens = obtain_target_lambda_sequence(
        target_tokens.squeeze(0), var_index_mask_no.squeeze(0), lambda_index_mask.squeeze(0), app_index_mask.squeeze(0))

    bos = target_embs[0, 0, :].unsqueeze(0).unsqueeze(0)
    seq_syntax = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) 
    
    target_tokens[target_tokens == -1] = 0

    outs = model(target_embs[:, :-1, :], seq_syntax[:, :-1], in_embs, sent_pad_mask) # get_discrete_output(in_embs, model, target_tokens.shape[1])
    if len(outs) == 3:
        out, classified_class, var_reg = outs
    else: 
        out, classified_class = outs
        var_reg = None
    target = target_embs[:, 1:, :]
    lambda_index_mask, app_index_mask, var_index_mask_no, stop_mask, pad_mask = (lambda_index_mask[:, 1:], app_index_mask[:, 1:], var_index_mask_no[:, 1:], stop_mask[:, 1:], pad_mask[:, 1:])
    #classiifer truth
    gt_cls_mask = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) #+ 4*stop_mask.type(torch.bool) #because lambda's class is 2
    loss = nn.functional.cross_entropy(classified_class.view(-1, 4), gt_cls_mask.view(-1), reduction="none")
    loss = ((loss) * (0.95 ** (torch.arange(loss.shape[0])).to(gt_cls_mask.device))).mean()# -- discounted loss

    return bos, loss, target, classified_class, var_reg, gt_cls_mask, in_embs, in_tokens, true_tokens

def model_inference(model, dataloader, 
                    max_len=200, last=False, 
                    beam_size=1, split="train",
                    word_beam=False,
                    stop_early=False,
                    stop_duplicates=False,
                    ret_token_indices=False):
    global DEVICE
    model.to(DEVICE)
    model.eval()
    confusion_matrix = torch.zeros(4, 4)
    average_loss = 0
    count = 0
    outs = []

    for_dist_out = []
    for_dist_out_ret = []
    skips = []

    prs, ps = [], []
    with torch.no_grad():
        pbar = tqdm.tqdm(total=min(100000, len(dataloader)))
        for k, batch in enumerate(dataloader):
            bos, loss, out, classified_class, var_reg, gt_cls_mask, in_embs, in_tokens, true_tokens = teacher_forcing(model, batch)

            if len(true_tokens) < 2: 
                skips.append(k)
                continue

            average_loss += loss.item()
            count += 1
            pbar.set_description(f"Loss: {loss.item()/count}")

            # #probability of true sequence:
            pr = torch.nn.Softmax(dim=-1)(classified_class)
            # pr = torch.gather(pr, -1, pr.argmax(-1,keepdim=True)).squeeze(-1) # teacher-forcing prob -- for true replace with gt_cls_mask
            pr = torch.gather(pr, -1, gt_cls_mask.unsqueeze(-1)).squeeze(-1) 
            
            #Write the written outputs:
            # outs.append([list(zip(gt_cls_mask.squeeze(0).tolist(), pr.squeeze(0).tolist())), get_out_list(out[0], classified_class[0], var_reg[0] if var_reg is not None else None, in_embs[0], in_tokens[0]), pr.prod(dim=-1).squeeze(0).item()])
            
            out_inf, classified_class_inf, var_reg_inf, probs_inf = model(in_embs, in_tokens, max_len=max_len, last=last, beam_size=beam_size, bos=bos, reference_target=out,
            word_beam=word_beam, stop_early=stop_early, stop_duplicates=stop_duplicates)

            if beam_size <= 1: 
                # outs.append([list(zip(classified_class_inf.argmax(-1).squeeze(0).tolist(), classified_class_inf.max(dim=-1)[0].squeeze(0).tolist())), got_seq, probs_inf.prod().item()])
                got_seq = get_out_list(out_inf[0], classified_class_inf[0], var_reg_inf[0] if var_reg_inf is not None else None, in_embs[0], in_tokens[0])
                for_dist_out.append([" ".join(true_tokens), got_seq])
                if ret_token_indices:
                    got_seq, got_indices = get_out_list(out_inf[0], classified_class_inf[0], var_reg_inf[0] if var_reg_inf is not None else None, in_embs[0], in_tokens[0], ret_token_indices=ret_token_indices)
                    for_dist_out_ret.append([got_seq, got_indices])
                p = probs_inf.prod().item()
            else:
                # p = -1
                # for jkl in range(len(classified_class_inf)):
                #     total_prob = probs_inf[jkl].prod().item()
                #     if total_prob > p:
                #         p = total_prob
                #     outs.append([list(zip(classified_class_inf[jkl].tolist(), probs_inf[jkl].tolist())), get_out_list(out_inf[jkl], classified_class_inf[jkl], var_reg_inf[jkl] if var_reg_inf is not None else None, in_embs[0], in_tokens[0]), total_prob])
                # outs.append(["", "", ""]) # emmpty divider

                ap_list = []
                for jkl in range((beam_size)):
                    total_prob = torch.gather(probs_inf[jkl], -1, classified_class_inf[jkl].unsqueeze(1)).squeeze(1).prod().item()
                    got_seq = get_out_list(out_inf[jkl], classified_class_inf[jkl], var_reg_inf[jkl] if var_reg_inf is not None else None, in_embs[0], in_tokens[0])
                    ap_list.append([" ".join(true_tokens), got_seq, probs_inf[jkl].detach().tolist(), total_prob])
                    if jkl == 0: p = total_prob
                for_dist_out.append(ap_list)

            prs.append(pr.prod().item())
            ps.append(p)
            pbar.update(1)
            if k>100000: break

    # #write
    loss = average_loss / count
    print("Average Loss: ", loss)

    # csv_file = open(f"{split}_outputs_gahhhh.csv", "w")
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerows(outs)
    # csv_file.close()

    #difference?
    to_plot = [int(prs[i] >= ps[i]) for i in range(len(prs))]

    plt.bar([0, 1], [len(to_plot) - sum(to_plot), sum(to_plot)])
    plt.xlabel('Size')
    plt.ylabel('Count')
    plt.title('Size Distribution')
    plt.savefig(f"outputs/{split}_all_lev_prs.png")

    if beam_size > 1:
        data = list(zip(*for_dist_out))
        print(len(data[0]), len(data[1][0]), len(data))
        for b in range(beam_size):
            for_dist_out = [l[:2] for l in data[b]]
            csv_file = open(f"outputs/{split}_all_lev_b{b}.csv", "w")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(for_dist_out)
            csv_file.close()
            for_dist_out = [l[2:] for l in data[b]]
            csv_file = open(f"outputs/{split}_all_lev_b{b}_probs.csv", "w")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(for_dist_out)
            csv_file.close()

    else:
        csv_file = open(f"outputs/{split}_all_lev.csv", "a")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(for_dist_out)
        csv_file.close()
        print(f"Wrote {len(for_dist_out)} lines to csv file")
        if ret_token_indices:
            pickled_f = open(f"outputs/{split}_all_lev.pkl", "wb")
            pickle.dump(for_dist_out_ret, pickled_f)
            pickled_f.close()

    skips_file = open(f"outputs/{split}_all_lev_skips.pkl", "wb")
    pickle.dump(skips, skips_file)
    skips_file.close()
    print(skips)
    # plot_teacher_forcing_error(prs, ps, save_as="teaching_forcing_not_last.png")
    # mean_probability_measures(word_prs, word_ps, title="Evolution of Word Probabilities", save_as="word_time_notlast.png")
    # mean_probability_measures(var_prs, var_ps, title="Evolution of Var Probabilities", save_as="var_time_notlast.png")
    # mean_probability_measures(lambda_prs, lambda_ps, title="Evolution of Lambda Probabilities", save_as="lmda_time_notlast.png")
    # mean_probability_measures(app_prs, app_ps, title="Evolution of App Probabilities", save_as="app_time_notlast.png")


    return confusion_matrix

if __name__ == "__main__":
    import os
    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--data_path", default="", required=True)
    parser.add_argument("--name", default="", required=True)
    parser.add_argument("--word_beam", action="store_true")
    parser.add_argument("--bert_type", required=True)
    parser.add_argument("--stop_early", action="store_true")
    parser.add_argument("--stop_duplicates", action="store_true")
    parser.add_argument("--ret_token_indices", action="store_true")
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--simplest", action="store_true")

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["BERT_TYPE"] = args.bert_type

    DEVICE = torch.device("cpu") if args.cpu else torch.device("cuda:0")

    assert os.environ["BERT_TYPE"] in SPEC_TOKENS
    print("---Model type is ", os.environ["BERT_TYPE"])

    #load model
    model = TransformerDecoderStack(4, 384, 8, 3072, custom=args.custom)
    checkpoint = torch.load(args.model_path)
    model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model_weights.update({k: torch.zeros_like(v) for k, v in model.state_dict().items() if k not in model_weights})
    model.load_state_dict(model_weights)

    model = InferenceModel(model)
    train_dataloader, valid_dataloader, test_dataloader = dataloader.data_init(1, last=args.last, inference=True, data_path=args.data_path, rem_spec_sentences=True, filtered=args.filtered, simplest=args.simplest)
    
    # sanity test
    # pbar = tqdm.tqdm(total=min(100000, len(test_dataloader)))
    # for k, batch in enumerate(test_dataloader):
    #     pbar.update(1)
    #     continue

    # model_inference(model, train_dataloader,
    #                 max_len=200, last=args.last,
    #                 beam_size=args.beam_size, split=f"{args.name}_train")
    # model_inference(model, valid_dataloader, max_len=200, 
    #                 last=args.last, beam_size=args.beam_size, 
    #                 split=f"{args.name}_valid")
    model_inference(model, test_dataloader, max_len=200,
                     last=args.last, beam_size=args.beam_size,
                       split=f"{args.name}_test",
                       word_beam=args.word_beam,
                       stop_early = args.stop_early, 
                       stop_duplicates=args.stop_duplicates,
                       ret_token_indices=args.ret_token_indices)

    # # model = ShuffledTransformerStack.load_from_checkpoint(args.model_path, model).model
    # DEVICE = torch.device("cpu") if args.cpu else torch.device("cuda")
    # model = model.to(DEVICE)

    # # --LOAD DATA
    # dataloader, valid_dataloader, test_dataloader = dataloader.data_init(1, last=args.last)

    # # --DISCRETE OUTPUT SAMPLES
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

#command:  python inference.py --model_path models/train_r3_varrg.ckpt --last --custom
# --model_path models/best_bert_base_r2.ckpt --cpu --last --custom --beam_size 4 --data_path /w/nobackup/436/lambda/bert_base/data/ --name beam_test_bert_base 

# --model_path models/best_roberta_base.ckpt --data_path /w/nobackup/436/lambda/roberta_base/data/ --cpu --last --custom --name roberta_test_beam --beam_size 4 --word_beam 

# --model_path models/best_roberta_base.ckpt --data_path /w/nobackup/436/lambda/roberta_base/data/ --cpu --last --custom --name roberta_base --bert_type roberta_base --stop_duplicates