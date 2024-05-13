from transformers import BertTokenizer, BertModel, BertConfig
import torch

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

BERT_MODEL = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

# BIG_VAR_EMBS = -torch.ones((2000, 768)) * (torch.tensor(range(1, 2001)))[:, None]

LAMBDA = [-3.3135e+00, -2.7586e+00, -3.6889e-01, -3.7267e+00,  3.3720e+00,
         8.9147e-01,  2.3475e+00, -2.5278e+00, -4.6680e-01, -4.7528e+00,
        -1.0770e+00,  2.9502e+00,  1.5172e+00, -1.8565e+00, -2.3205e-01,
         6.8677e-01,  3.2492e+00,  8.4883e-01,  6.5826e-01,  2.3877e+00,
         1.5019e+00,  1.7078e-01, -3.1180e+00,  2.8585e-01,  8.9036e-01,
         2.2260e+00,  2.4465e+00,  4.2735e+00, -4.2122e+00,  4.4266e+00,
         3.1142e+00,  2.1118e-01,  1.8732e+00,  1.0721e+00, -3.0133e+00,
         1.2542e-01,  2.1987e+00,  9.9317e-02, -2.5550e+00,  2.1955e+00,
         5.9998e-01, -4.2951e+00,  7.1179e-01, -2.9205e+00,  5.7692e-01,
         7.0469e-01,  1.8296e+00,  8.5169e-01, -3.3169e+00, -3.3497e+00,
        -7.3086e+00, -6.8566e-01, -3.3478e+00,  1.3948e+00, -1.4006e-01,
        -3.1756e+00,  9.0615e-01, -1.5308e+00, -2.2150e+00, -2.9371e+00,
        -1.1899e+00, -1.9889e+00,  2.4214e+00,  1.2941e+00,  8.1149e-01,
         1.0494e+00,  2.4698e+00, -1.1647e+00, -5.7892e+00,  3.1916e+00,
        -2.1611e+00,  3.0804e+00,  2.5953e+00,  3.1845e-01,  2.1278e+00,
         1.7139e+00,  6.5423e-01,  2.0100e+00,  1.4426e+00, -2.0651e+00,
         1.8783e+00,  3.7932e+00,  3.6327e+00,  1.6342e+00,  6.0972e-02,
         1.3107e+00, -2.9254e-01, -9.0741e-01, -1.0057e+00, -5.6712e-01,
        -2.3282e+00,  3.9747e-01, -3.1881e+00, -1.0846e+00,  1.8105e-01,
        -1.3181e+00, -3.8823e+00, -2.1056e-01, -5.7616e-02, -5.0431e+00,
        -2.6774e+00, -3.2328e+00,  3.4712e+00,  1.5128e+00, -3.1794e+00,
        -3.7099e-01, -3.2249e+00, -1.3762e+00, -5.7126e-02,  1.0611e+00,
        -1.2332e+00,  3.9637e+00,  1.8836e+00, -3.2353e-01, -2.2623e+00,
         4.8379e+00, -1.8501e+00, -1.5714e+00, -5.4840e-01,  5.1024e-01,
        -3.5155e-01, -2.5558e-01, -6.7164e-01,  2.1328e+00, -1.1403e+00,
        -2.1551e-01, -1.3572e+00, -9.0263e-01, -9.0670e-01, -5.1495e-01,
        -1.3668e-01,  2.3971e+00,  6.7695e-01,  9.7914e-01, -3.1027e+00,
        -2.7494e+00, -2.3876e+00, -2.7223e+00,  1.2180e+00, -3.8373e+00,
        -3.6043e-01,  2.7373e+00,  1.6424e+00, -4.2736e+00,  1.3567e+00,
        -8.1596e-01, -1.6842e+00, -1.6346e+00, -2.1938e-01, -2.0256e+00,
        -2.0207e+00,  2.6539e+00, -3.8748e-01, -2.1815e+00,  2.9348e-01,
        -6.7919e-01,  1.8233e-02,  2.5751e+00, -1.6672e+00,  3.2755e+00,
         4.0157e+00,  1.8416e+00,  6.1887e-01,  1.4130e-01,  5.6046e-01,
        -8.3224e-02,  2.1938e+00,  5.3669e+00, -2.2245e+00,  1.0204e+00,
        -2.7466e+00, -3.0379e+00,  4.3921e+00, -2.3160e+00, -4.1867e+00,
        -4.1303e-01,  2.3688e+00, -2.0843e+00,  4.3019e-01, -1.0586e+00,
        -1.6397e+00,  3.1751e-01, -2.5750e+00, -5.7433e+00,  5.3298e+00,
        -6.6908e-02,  3.2317e+00, -4.8599e+00,  4.3985e+00,  1.5649e+00,
        -2.8616e+00, -2.3253e+00, -1.2751e+00, -1.7033e+00,  1.2773e-01,
        -1.2860e+00,  1.1555e+00, -1.2353e+00,  1.4372e+00, -4.1583e-01,
         1.1137e+00,  1.2843e+00,  3.2876e-01,  1.1938e-02,  1.2445e+00,
         7.5473e-01,  2.6764e+00, -7.5150e-01, -1.6330e+00, -6.4558e-01,
        -1.9505e+00,  3.5371e+00, -4.6894e-01,  1.0590e+00,  2.4523e+00,
        -3.2503e+00,  1.6509e+00,  6.3387e-01, -2.8172e-01,  5.5682e-01,
         6.4716e-01, -4.5624e+00,  1.0307e+00,  1.9889e+00, -2.9133e+00,
         2.5735e+00,  3.7944e+00, -1.7040e+00,  2.4851e+00, -3.3799e+00,
        -7.1099e-01, -2.1800e+00, -1.2324e+00,  2.0857e+00, -4.9846e+00,
         6.1239e+00,  1.7388e+00, -2.0179e+00, -5.4812e-02, -1.2018e+00,
         2.7741e-03,  5.2229e-02,  2.6430e+00,  9.0932e-01, -1.3388e+00,
         6.2962e-01,  3.3737e+00, -2.8421e+00,  8.2524e-01,  6.2403e-01,
        -1.4764e+00, -6.0667e+00,  5.2454e-01,  5.6433e-01, -3.0463e-01,
        -4.7386e+00,  3.0809e+00, -3.3186e+00,  3.7477e+00,  3.1179e-01,
        -6.6792e-01, -7.0630e-01,  5.8900e+00, -4.2758e+00, -3.8875e-01,
         1.9103e+00,  1.7846e+00,  2.0741e-01,  7.8334e-01, -3.1455e+00,
        -3.3192e+00,  8.8402e-01, -3.0739e+00,  3.5844e-01, -3.1313e+00,
        -2.9457e+00,  3.5284e+00, -1.4172e+00, -1.6506e+00,  1.1599e+00,
         1.8039e+00, -2.6333e+00, -2.0823e+00, -9.0513e-01,  1.1016e+00,
        -2.8493e+00, -4.4306e+00, -7.1420e+00,  3.2037e-01, -3.3057e+00,
        -5.6570e-01,  4.2227e+00,  2.4256e+00, -4.2043e+00,  2.5048e+00,
         1.4394e+00, -2.0516e+00, -4.4857e-01, -1.1194e+00, -2.3096e+00,
        -3.9147e+00,  2.7798e+00,  1.0817e+00, -1.1964e+00,  1.3691e+00,
         4.2813e+00, -2.4028e+00,  3.7370e+00, -2.6184e+01, -6.7294e-02,
        -1.6096e+00, -2.9655e+00,  2.5904e+00,  4.8800e-01, -2.6602e+00,
         1.7182e+00, -8.7960e-01, -2.0941e+00,  5.6633e-01, -2.3083e+00,
        -2.6027e-01,  1.3904e+00,  9.6869e-01, -1.4293e+00,  3.7773e+00,
        -1.1938e+00, -8.4304e-01,  7.6389e+00, -2.9389e+00, -1.0067e+00,
         5.7023e+00,  7.6335e-01, -2.6566e+00, -4.8692e-02,  2.8855e+00,
         6.9106e-01, -4.6193e+00, -1.7560e+00, -2.9351e+00, -7.3184e-01,
        -8.4640e-01,  2.0984e+00,  1.4767e+00,  2.1162e+00,  2.9690e+00,
        -1.6619e+00,  5.1061e+00,  2.4482e+00,  1.9225e+00,  3.2702e+00,
        -1.3703e+00,  2.6375e+00,  6.3780e-01, -3.1983e+00,  3.0019e+00,
         2.2784e+00, -3.1191e-01,  5.1687e-02,  1.6991e+00,  9.3549e-01,
         9.8553e-02, -2.9478e+00,  1.1944e+00,  8.4290e-01, -7.7147e-01,
         1.2980e+00, -2.0717e+00, -5.7824e-01,  4.4750e+00, -1.4466e+00,
        -1.0605e+00,  6.0589e+00, -5.4886e+00,  2.7229e-01, -4.5031e+00,
        -5.8624e-01,  2.3225e+00, -1.4056e+00, -2.1504e-01, -2.4424e+00,
         3.2497e+00, -8.1334e+00, -6.2590e-01,  5.5696e-01, -4.8258e+00,
        -1.1730e-01,  1.9744e+00, -3.4267e+00,  1.3859e+00, -1.7556e+00,
        -1.1497e+00,  1.2288e+00,  1.3327e+00,  2.2444e+00,  4.7527e-01,
        -2.6762e+00, -1.8340e+00, -4.1191e-01, -2.0536e+00, -4.5965e-01,
         1.3303e+00,  1.6991e+00,  1.2802e+00, -2.3453e-01,  2.7563e+00,
        -1.5400e+00,  1.4303e+00,  2.4629e+00, -2.5018e+00, -1.7655e+00,
        -2.0716e+00,  3.3955e+00, -5.7036e-01, -1.8479e-01, -4.6444e+00,
         2.6608e+00, -1.4472e-01, -9.5336e-01,  2.7766e+00,  3.5584e-01,
         2.7575e+00,  1.1634e-01, -1.0449e+00, -8.0447e-01, -3.3874e-01,
        -1.9600e+00, -9.5284e-01,  3.6151e-02, -4.6078e+00,  1.8852e+00,
        -2.5373e+00, -9.1922e-01, -2.4230e+00,  1.2126e-01,  1.0512e+00,
        -3.6575e+00, -4.5906e+00,  3.5362e+00,  4.1398e-01, -3.8761e-02,
         2.7149e+00, -9.9391e-01, -4.0967e+00, -1.0162e+00, -2.0946e+00,
        -1.8470e+00, -7.0160e-01,  4.0804e+00,  3.6075e+00,  3.9183e-01,
        -1.5675e+00,  7.2158e-01,  2.8535e+00, -2.4977e-02,  4.1625e-01,
         1.9402e+00, -1.3949e+00,  5.9175e+00,  3.3740e+00, -2.6847e+00,
         3.4491e+00,  1.4475e+00, -2.8580e+00,  1.8266e+00, -8.3584e-02,
        -2.8492e+00,  1.5923e+00, -3.6305e+00,  2.5283e+00,  6.3843e-01,
         2.8111e+00,  8.8712e-02,  3.4710e+00,  3.5513e+00, -3.0167e+00,
         1.2969e+00, -8.2174e-01, -1.5517e+00,  4.0462e+00, -6.8676e-02,
         4.5384e-01,  3.7121e+00,  1.1209e+00, -2.3013e+00,  2.0804e+00,
        -5.2170e+00, -2.4825e+00,  1.5862e+00,  2.5573e+00,  2.5523e+00,
        -9.8962e-01, -8.5888e-01,  5.9046e-01,  1.0379e+00, -4.7854e+00,
        -4.3483e-01, -2.1167e-01,  3.6117e+00,  5.9636e-01, -3.5200e+00,
         1.3937e-01,  3.5062e-01, -9.8724e-01, -2.7051e-02,  2.7355e-01,
         3.4469e+00,  1.1308e+00,  1.7603e+00, -3.2232e+00, -8.0583e-01,
         1.2998e+00,  1.7400e-01, -1.4148e+00, -8.8007e-01,  1.5785e+00,
         3.8238e+00, -1.1245e+00, -1.6598e+00,  3.1723e+00, -1.3923e+00,
        -3.9705e-01, -2.7059e+00, -3.8049e+00,  4.1184e+00,  9.3736e-01,
        -1.9206e-01,  7.5266e-01, -6.1760e+00, -2.4342e+00, -3.2933e+00,
         1.7069e+00, -2.5233e+00,  1.2992e+00,  2.3805e+00, -3.0487e+00,
        -5.1008e+00,  2.7940e-01, -2.2093e-02, -1.6386e+00, -1.1237e+00,
        -1.2678e+00,  4.7244e-01,  3.2316e+00, -2.8693e+00, -5.5920e+00,
         2.7556e+00, -1.7640e+00, -1.4405e+00, -3.3584e-02, -3.8393e+00,
        -3.4522e+00,  3.6886e-01,  1.1239e+00,  6.2688e-01,  3.2945e-01,
         2.5892e-01,  6.4228e+00,  6.3221e+00, -4.5156e+00, -2.7281e+00,
        -9.1761e-01, -1.2566e+00,  3.2518e-01, -4.4484e+00,  1.2588e+00,
         5.9508e-01,  2.7892e-01,  1.7514e+00, -1.1102e+00,  3.6241e+00,
        -3.8306e-01,  1.5152e+00, -2.5707e+00,  1.2876e+00,  1.9059e+00,
         2.7567e+00,  3.1745e+00, -1.1340e-01, -6.2290e-01,  8.3767e-01,
         3.9998e-01,  2.4515e+00, -7.5721e-01, -1.1941e-01,  2.6214e+00,
        -1.5989e+00, -1.6428e-01,  1.0623e+00, -1.6508e+00,  4.4860e+00,
         1.4736e+00,  2.0480e-01,  2.5109e+00, -5.3497e-01, -1.3645e+00,
        -1.7797e+00,  3.5331e-01, -3.5680e+00,  1.8331e-01,  5.6777e-01,
        -2.9322e+00,  3.2359e+00,  3.1235e+00,  2.1699e+00, -2.3621e+00,
         3.0739e-01, -4.8564e+00, -2.6603e-01, -1.4907e+00,  7.3974e-01,
        -2.4895e+00, -5.7964e-01, -1.6662e+00,  3.1351e+00,  2.1481e+00,
         3.3474e+00,  1.0741e-01,  4.0312e+00, -1.4088e+00, -8.5506e-01,
        -2.2909e+00,  3.4207e+00, -3.9149e-01,  3.2170e-02, -1.2453e+00,
        -3.5334e+00,  2.6726e+00,  1.6469e+00, -8.1121e-01,  2.0912e+00,
         3.1388e+00,  1.7429e+00,  4.4104e+00, -1.1352e+00, -3.8583e+00,
        -5.2907e-01,  2.5554e-01,  3.1201e+00,  3.0946e+00, -4.5495e+00,
         1.1542e+00,  3.7441e+00,  1.3572e+00, -9.0323e-01,  3.4431e+00,
        -8.1582e-01, -3.2206e+00, -1.4905e+00, -3.8101e-01,  1.5171e+00,
         3.8524e-01, -2.6928e-02, -1.8746e+00, -2.0259e+00, -2.7127e+00,
        -3.3234e+00, -2.8643e+00,  5.3517e+00,  2.1286e+00, -1.3188e+00,
        -2.0497e+00,  3.6086e-01, -3.9129e-02, -5.2416e+00, -2.8535e+00,
         8.8734e-02, -2.5806e+00,  1.0068e-01,  2.7651e+00,  1.8681e+00,
         2.8915e+00,  1.3750e+00, -3.9307e-01, -4.0072e-01,  9.9097e-01,
        -1.3385e+00, -9.0931e-01, -4.9930e+00,  1.0940e+00,  2.3923e-01,
        -4.5788e-01,  2.4257e+00,  1.6875e+00, -2.4681e-01, -1.0035e+00,
         2.6910e+00, -4.5623e+00, -2.1969e+00,  5.3974e-01,  3.6894e-01,
         6.3868e-01,  2.0471e+00,  7.4379e+00, -4.1731e+00,  1.8261e+00,
         1.0422e+00,  1.3115e+00,  8.8956e-01,  2.2040e+00, -1.5828e+00,
        -1.0082e+00, -2.3505e+00,  3.1208e+00, -1.0809e-01,  7.4070e-01,
        -1.8415e-01,  3.5365e+00,  3.6353e-01,  8.6545e-01, -3.0903e-01,
        -5.4256e-01, -1.1284e+00, -4.1550e+00, -2.6998e-01, -6.8867e-01,
         4.1615e+00, -2.2660e+00, -5.4239e+00,  2.9983e+00,  1.3021e+00,
        -1.3561e+00,  2.0938e+00, -9.9152e-01, -1.2103e+00,  4.4951e-01,
        -1.8396e+00,  1.3715e+00, -1.0615e+00,  1.4246e-01, -2.0520e+00,
         5.9700e-02,  2.3642e+00, -1.6028e+00, -3.2759e+00, -4.5145e+00,
        -3.2357e-01, -6.2740e-01,  4.4916e+00,  1.5595e+00,  3.9803e+00,
        -2.8854e-01,  2.9452e+00, -8.5055e-01, -2.1731e+00,  9.3187e-01,
        -2.4485e+00,  6.5658e-01, -6.5460e-01,  1.2841e+00,  6.4110e+00,
        -4.0646e+00, -5.6506e+00,  9.3220e-01,  1.0059e-01, -2.4745e+00,
         5.1543e-01, -2.2713e+00, -5.4883e-01,  9.4346e-01, -1.8255e+00,
         3.4565e-01,  5.2624e-01,  7.6441e-01,  5.5507e-01,  8.3721e-01,
        -1.3733e+00,  5.3714e-01,  8.8101e-01]

def create_out_embeddings(sentences, lamda=False):
    #tokenize tha sentence 
    tokenized = TOKENIZER(sentences, 
                          return_tensors="pt", #return torch tensors
                          padding=True, #pad to max length in batch
                          truncation=True) #truncate to max model length
    #search for the tokenized id of λ
    input_ids = tokenized["input_ids"]
    lambda_id = TOKENIZER.convert_tokens_to_ids("λ")

    if lamda:
        #get all the indices where the lambda token is presen
        lambda_index_mask = (input_ids == lambda_id)
        var_index_mask = (input_ids == TOKENIZER.convert_tokens_to_ids("np"))
        var_index_mask = var_index_mask | (input_ids == TOKENIZER.convert_tokens_to_ids("##np"))
        #shift the 1s in var_index_mask to the right by 1
        var_index_mask_underscore = torch.roll(var_index_mask, shifts=1, dims=1)
        #roll again
        var_index_mask_no = torch.roll(var_index_mask_underscore, shifts=1, dims=1)
        var_index_mask_no = torch.where(var_index_mask_no == 1, input_ids, 0) #make the variable numbers in the mask
        #to ensure uniqueness batch wise we add a constant along the batch dimension
        var_index_mask_no_new = var_index_mask_no + torch.arange(0, var_index_mask_no.size(0)).reshape(var_index_mask_no.size(0), 1)
        #everything that isnt var no shud be zero still
        var_index_mask_no_new[var_index_mask_no == 0] = 0
        var_index_mask_no = var_index_mask_no_new

        _, var_index_mask_no = torch.unique(var_index_mask_no, return_inverse=True) # contiguous naming


    # the mask of tokens belonging to the variable name, both next to the lambda and within an expression
    pad_mask = (input_ids == TOKENIZER.pad_token_id)
    return (tokenized, lambda_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask) if lamda else (tokenized, pad_mask)

def get_bert_emb(tokenized_sents):
    #get the bert embeddings
    global BERT_MODEL
    BERT_MODEL = BERT_MODEL
    with torch.no_grad():
        outputs = BERT_MODEL(**tokenized_sents, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-4:]
        #sum the last four hidden states
        embs = torch.stack(hidden_states, dim=0).sum(dim=0)
    #move BERT back to cpu
    BERT_MODEL = BERT_MODEL
    return embs.detach() # no grads through bert ever

def process_bert_lambda(tokenized_sents, lambda_index_mask, var_index_mask, lambda_norm=True, var_norm=True):
    assert lambda_norm if var_norm else True, "norm_lambda cant be off and norm_var be on"
    # global BIG_VAR_EMBS
    #get the bert embeddings
    var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = var_index_mask
    embs = get_bert_emb(tokenized_sents)
    #time to mask out the variables
    if var_norm: 
        embs[var_index_mask | var_index_mask_underscore | pad_mask] = torch.zeros_like(embs[0, 0, :]) # also make hte pad embeddings 0
        mask_sort = torch.argsort((var_index_mask | var_index_mask_underscore).to(torch.uint8), stable=True) #move the embeddiungs to the end
        embs = torch.gather(embs, -1, mask_sort.unsqueeze(-1).expand(-1, -1, embs.size(-1)))  # all the var names and the underscores have been moveed to the end
        
    #     #now we have the var_numbers which we need to uniq-ify
    #     uniques, indices = torch.unique(var_index_mask_no.reshape(-1), return_inverse=True, sorted=True)
    #     # mapping = torch.arange(1, len(uniques) + 1, dtype=torch.int64)
    #     # new_var_no_index = mapping[indices]
    #     # new_var_no_index = new_var_no_index.reshape(var_index_mask_no.shape)
    #     # embs[var_index_mask_no != 0] = BIG_VAR_EMBS[var_index_mask_no != 0]

    #     # embs[var_index_mask_no != 0] = BIG_VAR_EMBS[indices != 0]
    #     #one time operation, perform on CPU
    #     # indices = indices.to(torch.device('cpu'))
    #     # var_index_mask_no = var_index_mask_no.to(torch.device('cpu'))
    #     embs = embs.index_put((var_index_mask_no != 0, ), BIG_VAR_EMBS[indices[indices != 0]], accumulate=True)
    if lambda_norm:
        embs[lambda_index_mask] = torch.tensor(LAMBDA, device=embs.device, dtype=embs.dtype)#torch.ones((embs.shape[-1], ), device=embs.device, dtype=embs.dtype)
    #back to gpu
    return embs

def process_bert_word_lambda(tokenized_sents, lambda_index_mask, var_index_mask, lambda_norm=True, var_norm=True):
    #word embeddings instead of contextual embeddings
    assert lambda_norm if var_norm else True, "norm_lambda cant be off and norm_var be on"
    pass



