from transformers import BertTokenizer, BertModel
import torch

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

def create_in_embeddings(sentences):
    #tokenize tha sentence 
    tokenized = TOKENIZER(sentences, return_tensors="pt", padding="max_length", truncation=True)
    #search for the tokenized id of λ
    input_ids = tokenized["input_ids"]
    lambda_id = TOKENIZER.convert_tokens_to_ids("λ")

    #get all the indices where the lambda token is presen
    lambda_index_mask = (input_ids == lambda_id)
    var_index_mask = (input_ids == TOKENIZER.convert_tokens_to_ids("np"))
    #shift the 1s in var_index_mask to the right by 1
    var_index_mask_underscore = torch.roll(var_index_mask, shifts=1, dims=1)
    #roll again
    var_index_mask_no = torch.roll(var_index_mask_underscore, shifts=1, dims=1)
    var_index_mask = var_index_mask | var_index_mask_underscore | var_index_mask_no

    return tokenized, lambda_index_mask, var_index_mask

def process_bert_out(bert_outs, lambda_index_mask, var_index_mask):
    pass