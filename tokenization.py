from transformers import BertTokenizer, BertModel, BertConfig
import torch

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

BERT_MODEL = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

BIG_VAR_EMBS = -torch.ones((200, 768)) * (torch.tensor(range(1, 201)))[:, None]

def create_out_embeddings(sentences):
    #tokenize tha sentence 
    tokenized = TOKENIZER(sentences, 
                          return_tensors="tf", #return torch tensors
                          padding="max_length", #pad to max length in batch
                          truncation=True) #truncate to max model length
    #search for the tokenized id of λ
    input_ids = tokenized["input_ids"]
    lambda_id = TOKENIZER.convert_tokens_to_ids("λ")

    #get all the indices where the lambda token is presen
    lambda_index_mask = (input_ids == lambda_id)
    var_index_mask = (input_ids == TOKENIZER.convert_tokens_to_ids("np"))
    #shift the 1s in var_index_mask to the right by 1
    var_index_mask_underscore = tf.roll(var_index_mask, shifts=1, dims=1)
    #roll again
    var_index_mask_no = tf.roll(var_index_mask_underscore, shifts=1, dims=1)
    var_index_mask_no = torch.where(var_index_mask_no == 1, input_ids, 0) #make the variable numbers in the mask

    # the mask of tokens belonging to the variable name, both next to the lambda and within an expression
    pad_mask = (input_ids == TOKENIZER.pad_token_id)
    return tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask

def get_bert_emb(tokenized_sents):
    #get the bert embeddings
    with torch.no_grad():
        outputs = BERT_MODEL(**tokenized_sents, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-4:]
        #sum the last four hidden states
        embs = torch.stack(hidden_states, dim=0).sum(dim=0)

    return embs
def process_bert_lambda(tokenized_sents, lambda_index_mask, var_index_mask, lambda_norm=True, var_norm=True):
    assert lambda_norm if var_norm else True, "norm_lambda cant be off and norm_var be on"
    #get the bert embeddings
    var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = var_index_mask
    embs = get_bert_emb(tokenized_sents)
    #time to mask out the variables
    if var_norm: 
        embs[var_index_mask | var_index_mask_underscore | pad_mask] = torch.zeros_like(embs[0]) # also make hte pad embeddings 0
        mask_sort = torch.argsort(var_index_mask | var_index_mask_underscore) #move the embeddiungs to the end
        embs = embs[:, mask_sort, :] # all the var names and the underscores have been moveed to the end
        
        #now we have the var_numbers which we need to uniq-ify
        embs[var_index_mask_no] = BIG_VAR_EMBS[var_index_mask_no]
    if lambda_norm:
        embs[lambda_index_mask] = torch.zeros_like(embs.shape[-1])

def process_bert_word_lambda(tokenized_sents, lambda_index_mask, var_index_mask, lambda_norm=True, var_norm=True):
    #word embeddings instead of contextual embeddings
    assert lambda_norm if var_norm else True, "norm_lambda cant be off and norm_var be on"
    pass




