import matplotlib.pyplot as plt
from models import *
import dataloader
import tqdm, csv, random, pandas as pd
from tokenization import get_bert_emb, TOKENIZER
from dataloader import SEP_TOKEN

import torch 
import torch.nn as nn


#model inference
def model_inference(model, dataloader):
    global DEVICE
    model.eval()
    confusion_matrix = torch.zeros(4, 4)
    with torch.no_grad():
        #prpgress bar
        for batch in tqdm.tqdm(dataloader):
            (in_embs, target_embs, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask) = batch
            #move to device
            in_embs, target_embs, var_index_mask_no, lambda_index_mask, app_index_mask, pad_mask = in_embs.to(DEVICE), target_embs.to(DEVICE), var_index_mask_no.to(DEVICE), lambda_index_mask.to(DEVICE), app_index_mask.to(DEVICE), pad_mask.to(DEVICE)

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

def get_discrete_output(input_sent, model):
    model.eval()
    #bert tokenize and get embeddings
    tokenized = TOKENIZER(input_sent, return_tensors="pt", padding=True)
    input_embs = get_bert_emb(tokenized)
    input_embs = input_embs.to(DEVICE)
    #model inference
    out, classified_class, var_reg = model(input_embs)
    classified_class = classified_class.argmax(dim=-1).squeeze(0)
    var_reg = var_reg.squeeze(0)
    out_list = TOKENIZER.convert_ids_to_tokens(out)
    lambda_indices, app_indices, var_indices  = torch.where(classified_class == 2), torch.where(classified_class == 3), torch.where(classified_class == 1)
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
                sim = torch.cosine_similarity(var_reg[i], var_dict[key])
                if sim > min_sim:
                    min_sim = sim
                    min_key = key
            if min_sim > 0.9:
                out_list[i] = f"x{min_key}"
            else:
                var_dict[len(var_dict)] = var_reg[i]
                out_list[i] = f"x{len(var_dict)}"
    return " ".join(out_list)

class ClassifierModel(nn.Module):
    def __init__(self, model, linear):
        super().__init__()
        self.model = model
        self.linear = linear
    
    def forward(self, in_embs):
        x = torch.tensor(SEP_TOKEN).to(in_embs.device)
        out_stacked, classified_class_stacked, var_reg_stacked, newest_out = None, None, None, None
        list_out = []
        while newest_out != 102:
            out, classified_class, var_reg = self.model(x, in_embs)
            if out_stacked is None:
                out_stacked, classified_class_stacked, var_reg_stacked = out, classified_class, var_reg
            else:
                out_stacked = torch.cat([out_stacked, out], dim=1)
                classified_class_stacked = torch.cat([classified_class_stacked, classified_class], dim=1)
                var_reg_stacked = torch.cat([var_reg_stacked, var_reg], dim=1)
            newest_out = self.linear(out[0, -1, :]).argmax(-1)
            list_out.append(newest_out)
            x = out_stacked
        return list_out, classified_class_stacked, var_reg_stacked

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

    linear = nn.Linear(768, TOKENIZER.vocab_size)
    linear_weights = {k.replace("linear.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("linear.")}
    linear.load_state_dict(linear_weights)

    model = ClassifierModel(model, linear)

    # model = ShuffledTransformerStack.load_from_checkpoint(args.model_path, model).model
    DEVICE = torch.device("cpu") if args.cpu else torch.device("cuda")
    model = model.to(DEVICE)

    # --LOAD DATA
    # dataloader = dataloader.data_init(70, shuffled=True, mode=3)


    # -- CONFUSION MATRIX --
    # confusion_matrix = model_inference(model, dataloader)
    # plot_confusion_matrix(confusion_matrix)

    # --DISCRETE OUTPUT SAMPLES
    lines = pd.read_csv("data/input_sentences.csv", header=None)
    out_file = open("data/output_samples.csv", "w")
    out_file_csv = csv.writer(out_file)

    rand_lines = random.choices(range(len(lines)), k=10)
    write_lines = []
    for rand_line in rand_lines:
        line = eval(lines.iloc[rand_line, 1])
        words = " ".join(line).replace("...}"," ...}").replace("{..","{. .").replace("NP.","NP .").replace("NP—","NP —").replace(",}"," ,}").\
    replace("'re"," 're").replace("'s"," 's").replace("'ve}"," 've}").replace("!}"," !}").replace("?}"," ?}").replace("n't"," n't").\
    replace("'m}"," 'm}").replace("{. ..","{...").replace("{——}","{— —}").replace("{--—}","{- -—}").replace("St.", "St").split()
        
        words = [word[:-1] for i, word in enumerate(words) if i % 2 != 0]
        words = " ".join(words)

        write_lines.append([rand_line, get_discrete_output(words, model)])
    out_file.writelines(write_lines)




