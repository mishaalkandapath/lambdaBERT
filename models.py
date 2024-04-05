import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import tokenization

class TransformerDecoderStack(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.2, device="cpu"):
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dropout=dropout, device=device)
        self.decoders = nn.TransformerDecoder(decoder_layer, num_layers)
        self.d_model = d_model
        self.num_layers = num_layers
        self.final_forward = nn.Linear(d_model, d_model, device=device)
    
    def forward(self, seq, emb):
        out = self.decoders(seq, emb)
        return out
    
def train(model, train_loader, val_loader, epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for i, (seq, target_seq) in enumerate(train_loader):
            #tokenize the sqequences
            tokenized, in_pad_mask = tokenization.create_out_embeddings(seq)
            target_tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = tokenization.create_out_embeddings(target_seq, lamda=True)

            #get the bert embeddings
            in_embs = tokenization.get_bert_emb(tokenized)
            #mask out the pads:
            in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

            #get the bert embeddings for the target sequence
            target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, var_index_mask, lambda_norm=True, var_norm=True)
            #in_embs will be our encoder embs
            out = model(target_embs[:, :-1, :], in_embs)
            target = target_embs[:, 1:, :]

            #no mse for the pads, variables, lambda or anything else. jus tthe actual embeddings
            #offset the masks by one 
            lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no = (lambda_index_mask[:, 1:], var_index_mask[:, 1:], var_index_mask_underscore[:, 1:], var_index_mask_no[:, 1:])
            loss = criterion(out[~(lambda_index_mask | var_index_mask_no.astype(torch.bool) | pad_mask)],
                            target[~(lambda_index_mask | var_index_mask_no.astype(torch.bool) | pad_mask)])
            
            #take all lambdas into account and compute their difference with the first lambda #TODO: if this doesnt work, try a random perm pairing difference
            loss += criterion(out[lambda_index_mask].reshape(-1, out.size(-1))[0:1, :], out[lambda_index_mask].reshape(-1, out.size(-1))[1:, :])

            #contrastive losses on the variables at some point?
            #get the variables only :
            out_vars, target_vars = out[var_index_mask_no.astype(torch.bool)], target[var_index_mask_no.astype(torch.bool)]

            #get the first indices of the variables involved. 
            flattened_var_mask = var_index_mask_no[var_index_mask_no != 0]
            _, reverse_indices, counts = torch.unique(flattened_var_mask, return_index=True, return_counts=True)
            ind_sorted = torch.argsort(reverse_indices, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0]), cum_sum])
            first_indices = ind_sorted[cum_sum]
            
            #mseloss on this
            target_vars = out[first_indices][reverse_indices] #reference embeddings arrangement
            correct_nos = torch.count_nonzero(torch.count_nonzero(out_vars - target_vars, axis=-1) == 0)
            loss += criterion(out_vars, target_vars.detach())
            
            #count var var mismatches
            out_vars = out_vars.unsqueeze(1) - out_vars.unsqueeze(0)
            #count the number of zeros here
            nos = torch.count_nonzero(torch.count_nonzero(out_vars, axis=-1) == 0) - correct_nos

            loss += nos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for i, (seq, target_seq) in enumerate(val_loader):
                tokenized, in_pad_mask = tokenization.create_out_embeddings(seq)
                target_tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = tokenization.create_out_embeddings(target_seq, lamda=True)

                in_embs = tokenization.get_bert_emb(tokenized)
                in_embs[in_pad_mask] = torch.zeros_like(in_embs[0, 0, :])

                target_embs = tokenization.process_bert_lambda(target_tokenized, lambda_index_mask, var_index_mask, lambda_norm=True, var_norm=True)
                out = model(target_embs[:, -1, :], in_embs)
                target = target_embs[:, 1:, :]
                loss = criterion(out, target)

                print(f"Epoch {epoch} Iteration {i} Loss: {loss.item()}")
    return model

