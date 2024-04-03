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
            out = model(target_embs[:, -1, :], in_embs)
            target = target_embs[:, 1:, :]
            loss = criterion(out, target)

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

