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
from utils import ThreadLockDict

#global variables
thread_locked_dict = ThreadLockDict()

SAMPLE = [-4.2103e-01, -3.0598e-01, -2.2508e-01,  1.4646e-01,  5.2509e-02,
         4.2700e-01, -4.6492e-02,  1.3889e-01, -1.2491e-01, -4.8707e-01,
         3.8139e-01, -1.8956e-01,  5.7379e-01,  1.7551e-01,  2.2349e-01,
        -2.8812e-01,  3.3288e-01, -1.1145e-01,  2.1864e-01,  3.8258e-01,
        -2.9671e-01, -4.4660e-01, -3.0242e-01,  5.0644e-01,  7.5179e-01,
        -4.8174e-01,  8.8063e-02, -1.7766e-01, -2.3630e-01, -3.7674e-01,
         2.4552e-01,  5.0281e-01, -8.3439e-02,  2.0990e-01,  1.2476e-01,
         7.0918e-01,  2.4905e-01,  3.0341e-01, -5.7229e-01,  2.4940e-01,
         3.1851e-01, -1.1145e-01, -1.2311e-01, -4.8712e-01, -1.0294e-01,
        -2.7729e-01,  6.8541e-02, -1.4693e-01, -5.6295e-01,  5.7348e-01,
        -3.4611e-01, -6.2715e-02,  5.1864e-01,  7.0000e-01, -2.8143e-01,
        -4.2081e-04,  5.4147e-01,  2.8906e-02,  4.4382e-01, -2.7184e-02,
         6.0971e-01,  2.6835e-01,  8.0597e-04, -3.5461e-01, -1.5866e-01,
        -6.4282e-02, -1.1410e-01, -8.3978e-02, -1.1581e-02,  3.0003e-01,
        -4.9485e-02, -1.5263e-01,  1.1015e-01,  4.7283e-01, -1.8102e-01,
         5.2267e-01,  2.3820e-01, -3.2193e-01, -1.2074e+00, -5.5588e-02,
        -3.7953e-02,  6.1600e-01,  8.4649e-02,  1.8163e-01, -4.8665e-01,
         6.7326e-01,  9.8452e-02, -1.9982e-01, -2.3904e-01, -6.5457e-01,
        -4.0057e-01, -7.5356e-02,  1.9430e-01,  2.4619e-01, -3.7927e-02,
         4.6977e-01, -4.3362e-01, -2.3266e-01, -3.5312e-01, -6.8275e-03,
         2.0567e-02,  6.3186e-02,  4.8441e-01, -7.0720e-01,  5.7405e-01,
         4.7945e-01, -2.8358e-02, -3.8638e-01,  5.8037e-02,  1.7346e-01,
         3.2518e-01, -1.8407e-01,  3.3457e-01,  1.6588e-01,  3.8809e-01,
        -5.8488e-01, -1.9522e-01,  1.8143e-02, -2.0522e-01, -4.7781e-01,
        -2.9509e-01, -4.8812e-01,  2.0874e-01,  5.8320e-02,  3.4252e-01,
        -1.8311e-01,  5.9941e-01, -2.1832e-01,  4.3957e-01, -9.2462e-01,
         6.1281e-02,  3.0889e-01, -2.2607e-01,  4.1301e-01, -2.2773e-01,
         4.5186e-03,  3.5387e-01,  2.9703e-01,  2.3228e-01, -3.5043e-01,
        -1.5635e-01,  5.8049e-01, -6.1700e-02, -1.0873e-01,  4.0921e-03,
         4.6015e-01,  3.8612e-01,  4.7119e-01, -1.7915e-01,  2.8056e-01,
         4.4664e-01, -1.9231e-01, -7.3776e-02,  2.4211e-01, -7.2482e-01,
        -2.8597e-01,  7.7501e-01,  1.2067e-01, -3.9867e-01, -2.0801e-01,
        -5.3465e-02, -3.1983e-01, -3.9036e-01,  4.8856e-01,  3.7579e-01,
        -5.6263e-01, -1.7151e-01,  4.2882e-02,  9.3216e-02, -2.8277e-01,
        -3.9523e-01,  2.0320e-01,  7.4142e-03,  9.7085e-02, -3.9702e-01,
        -8.0566e-01,  7.4921e-02,  2.9620e-01,  3.2452e-01, -5.4641e-01,
        -6.1138e-01,  3.4696e-01,  4.2358e-01, -8.9184e-03, -1.0243e-01,
         8.4498e-02,  1.0015e-01, -2.4397e-01, -1.2899e-01,  6.6327e-01,
        -2.3303e-01,  2.1023e-01,  5.7512e-01, -7.9110e-01, -1.2046e-01,
         5.6683e-01,  3.2698e-01, -2.9426e-01, -2.0326e-01,  3.3325e-01,
         6.2946e-02,  5.7783e-02,  1.7142e-01,  1.0742e-01, -4.7378e-02,
         7.9864e-01,  4.9144e-01,  1.2202e-01,  4.0004e-01,  1.1955e+00,
        -1.4682e-01,  4.3441e-01,  6.5618e-01,  9.4515e-02,  5.9149e-01,
         4.7429e-01, -2.8162e-01, -2.3796e-01,  7.6754e-01,  3.3437e-01,
         3.6084e-01,  4.7911e-01, -4.8327e-01, -3.2655e-01, -6.3572e-03,
        -7.1023e-01,  5.9391e-01,  3.1122e-01, -1.4405e-01, -2.7487e-01,
        -2.0141e-02, -1.7073e-01, -4.6742e-02,  2.9522e-01, -2.5853e-02,
         8.9839e-02,  6.5251e-01,  3.0955e-02, -4.5042e-01, -1.7295e-01,
         5.3040e-02, -2.9064e-01,  4.7768e-02, -3.7160e-02,  5.8767e-02,
         1.1493e-01, -3.3490e-01, -4.4602e-01,  1.8002e-01, -2.9676e-01,
         7.9031e-02,  2.2113e-01,  1.2475e-01, -3.4819e-01,  1.5805e-01,
        -2.0085e-01,  3.2790e-01, -2.2891e-01,  3.9438e-01, -3.8146e-01,
        -4.7773e-01, -7.2869e-01, -1.8250e-01,  4.6296e-01, -1.7444e-01,
        -3.3302e-02, -6.1264e-01,  2.3482e-01, -3.6360e-01,  1.9610e-01,
        -3.7349e-01,  4.0816e-01, -1.8574e-01,  1.3069e-01,  3.4180e-01,
         1.8530e-01, -2.9706e-01,  3.6959e-01, -4.2788e-01,  1.2981e-01,
        -1.5590e-01, -3.1138e-01,  2.1318e-01,  3.3712e-01, -3.3021e-01,
         1.3519e-02,  5.0266e-01, -3.9136e-01,  2.7808e-01, -4.0208e-01,
        -7.8483e-01, -3.9444e-01,  7.2291e-02, -1.8064e-01,  6.2287e-01,
         9.7389e-02, -5.4530e-01, -9.3371e-02, -7.3288e-02, -3.5475e-01,
         3.7901e-01,  2.9352e-01, -3.4047e-01,  4.3691e-01, -6.2074e-02,
         2.3541e-01, -4.7341e-01, -3.3393e-01,  5.9500e-01, -1.1805e-01,
        -4.1076e-01,  7.3699e-02, -3.5099e-01,  3.3213e-01,  2.3219e-01,
         7.2563e-02, -1.5228e-02, -2.0401e-01, -5.2083e-01,  1.0344e-02,
        -4.2382e-01,  1.8156e-01, -1.2549e-01,  5.5129e-01,  4.4505e-01,
         4.9841e-01,  1.8415e-03,  1.6927e-01, -4.3645e-01,  9.1714e-03,
        -2.8722e-02,  6.2537e-01,  1.4026e-04,  6.3764e-02, -1.8035e-01,
         3.3780e-02,  9.1185e-02, -1.2796e-01, -5.0653e-01, -9.8428e-02,
         7.3566e-02, -7.3001e-01,  7.0216e-01,  8.2545e-02,  9.4910e-01,
        -1.9797e-01, -1.3895e-01, -4.5449e-01,  5.4770e-01,  5.6425e-03,
         3.9596e-01,  3.8191e-01,  2.4230e-02, -1.0527e-01, -1.1994e-01,
         1.2346e-01, -8.3646e-01,  7.7966e-02,  6.4783e-02,  9.9516e-02,
         2.9277e-01,  1.0407e-01, -5.5225e-01, -1.3486e-01,  4.5617e-02,
         3.5569e-01,  7.7834e-01, -7.5492e-01, -2.2862e-01, -5.3326e-01,
         5.2411e-01,  1.1843e+00, -8.5557e-01, -1.3025e+00,  3.9474e-01,
         9.5381e-02, -5.9482e-01, -1.7526e-01,  5.7118e-01,  2.7578e-01,
        -1.2385e-01,  1.3201e-01,  4.5171e-01,  9.4290e-02,  6.1930e-02,
         1.6339e-02,  7.7638e-01,  1.8744e-01,  8.9279e-02, -9.3362e-02,
        -3.0638e-01,  1.3208e-01,  1.5909e-01,  2.3224e-01, -1.1117e-01,
         3.8748e-01,  3.5298e-01, -2.7463e-01, -6.7552e-01,  3.1718e-01,
         8.4664e-02, -2.3297e-01,  5.5668e-02,  1.0715e-01, -2.2351e-01,
         3.0531e-01,  2.7564e-01,  1.2071e-01, -3.2141e-01,  9.5779e-02,
         2.8951e-01,  5.1061e-01,  3.1371e-01,  2.9017e-01,  3.9376e-02,
         3.2189e-01, -5.1156e-01,  5.6076e-01,  9.1313e-02,  2.3476e-02,
         4.4282e-02,  3.7163e-01,  6.4869e-02, -8.1690e-01, -1.6010e-01,
        -2.2210e-01,  3.9146e-03,  3.2296e-01, -3.1117e-01,  1.3278e-01,
        -9.1562e-01, -6.4557e-02,  4.2978e-01,  3.9153e-01,  1.5298e-01,
         1.0312e-01, -1.2637e-01,  1.8401e-01, -6.2828e-01, -8.4540e-02,
        -5.4583e-02,  3.8988e-01,  1.0562e-01,  2.0610e-01, -5.4917e-01,
        -4.5154e-01,  1.2553e-01, -5.7033e-01, -2.2482e-01, -6.1803e-02,
        -2.5681e-02, -1.5130e+00, -5.8471e-01,  3.2851e-01, -2.3368e-01,
        -5.2462e-01, -5.3381e-01,  8.6380e-02, -5.6274e-02,  3.1795e-01,
        -7.4702e-02,  6.1393e-01, -3.5107e-01, -6.2126e-01, -2.8101e-01,
        -2.7779e-01,  1.6517e-01, -5.8916e-01, -5.9875e-02,  1.0186e+00,
        -5.9706e-01,  2.8038e-01,  2.4418e-01,  2.8350e-01, -7.2995e-01,
         6.9187e-02,  6.3953e-01, -1.8892e-01, -3.6790e-01, -1.1474e-01,
         2.0309e-01,  1.8405e-01,  5.4548e-01, -8.1062e-01, -3.4370e-01,
        -1.6707e-01,  1.9157e-01, -3.6644e-01,  2.4023e-01,  6.1730e-02,
         4.3119e-01, -6.3807e-01,  5.6524e-01, -1.3233e+00, -4.7004e-01,
        -8.2783e-02, -4.3952e-01, -5.4501e-01, -5.0879e-01,  1.1742e+00,
        -4.9629e-01,  7.5262e-01, -3.3566e-01, -1.4287e-02, -6.4138e-01,
         7.0257e-03, -8.9987e-02,  5.8859e-01,  1.7992e-01,  6.4888e-02,
        -1.6463e-01, -9.4216e-01,  1.3830e-01,  1.9822e-01, -4.7593e-01,
         4.5452e-02,  2.0421e-01,  8.8795e-02,  3.9993e-01,  1.7590e-01,
        -2.8094e-02,  9.3914e-01,  5.6308e-02, -3.5840e-01, -2.5909e-01,
         1.8397e-01, -5.0384e-01, -2.6168e-01, -1.9416e-01, -1.8908e-01,
        -6.3524e-01,  3.2371e-01, -3.2599e-01,  2.5504e-01,  2.0112e-01,
         6.1060e-01, -2.9103e-01, -4.0965e-01, -1.3575e-01,  2.7573e-01,
         5.7377e-02, -1.9602e-01,  2.2117e-01, -1.3918e-01, -6.9501e-02,
         8.1222e-01, -4.5807e-01,  6.1700e-01, -9.0604e-01, -1.4885e-01,
        -2.6885e-01, -3.7095e-01,  1.3869e-01, -1.1160e-01, -4.0128e-01,
         2.5875e-01,  4.2263e-01, -6.3628e-01, -2.6977e-01,  7.7944e-02,
        -2.5615e-01, -1.5989e-01, -8.8311e-02, -3.3153e-01,  3.2962e-01,
        -4.0525e-02, -7.4097e-01, -2.0786e-01,  3.8255e-02,  5.0091e-01,
         1.8637e-01,  2.5052e-01, -2.1249e-01, -3.4466e-01,  2.0939e-01,
         2.0708e-01, -5.6828e-01, -1.4477e-01,  1.8582e-01, -4.3289e-01,
        -9.7458e-02, -4.0359e-01,  2.4749e-01, -1.9247e-01, -3.7060e-01,
        -9.6489e-02, -3.4278e-02,  5.5957e-01,  2.1104e-02, -2.2972e-01,
         5.8105e-02,  3.8643e-01, -4.1189e-01, -5.6503e-02,  5.8881e-01,
         3.4027e-01,  5.4384e-01, -5.5456e-01,  1.5185e-01,  3.4879e-01,
         3.1540e-01, -2.8121e-01,  1.6673e-02, -5.0198e-01, -9.0985e-01,
         1.4775e-01,  4.0530e-01,  4.3260e-01,  2.3647e-02, -7.1910e-01,
         2.4540e-01, -4.3704e-01, -2.0070e-01, -4.0606e-01, -2.1610e-01,
        -2.0408e-01, -8.7146e-02,  4.2893e-01,  3.7569e-01,  6.7838e-01,
        -1.2343e-01,  3.5381e-01,  5.6919e-01, -4.4323e-01, -2.7056e-01,
        -1.3865e-01, -4.4183e-02, -2.2686e-01,  4.6878e-02, -2.1432e-01,
        -4.6179e-01,  1.0180e-01,  4.1586e-01, -1.6756e-01, -1.7053e-01,
         7.0253e-01, -1.9636e-01, -5.2260e-01,  2.1853e-01, -6.8631e-01,
        -5.4129e-02, -4.3291e-01,  8.8239e-02, -6.3006e-02,  3.9996e-02,
        -2.5565e-01, -2.8305e-01,  3.2620e-01, -7.5125e-01,  1.1920e-02,
         7.6763e-01, -4.9103e-01,  4.8125e-01, -1.9842e-01,  7.7935e-02,
         8.6192e-01,  5.5255e-01, -2.6914e-01, -1.3348e-01,  4.4880e-01,
         1.9934e-01,  4.6080e-01,  3.5741e-02,  1.9557e-01,  1.8289e-02,
         2.7674e-01, -5.2763e-01, -3.6778e-01, -1.7578e-02, -7.4149e-02,
        -2.4428e-01, -1.7928e-01,  3.5426e-01, -2.0554e-01, -2.5641e-01,
        -4.0580e-01, -3.5725e-01,  4.4715e-01, -3.7157e-01, -8.2575e-02,
        -2.2716e-01,  6.2849e-01,  3.3006e-02, -2.7474e-01, -4.1911e-02,
         6.9686e-02,  1.6740e-01,  2.8803e-02, -3.1872e-01,  3.2710e-01,
         2.1385e-01,  2.1940e-01,  2.0773e-03, -2.0162e-01, -5.3656e-02,
        -5.7710e-02,  5.4900e-02,  1.3739e-01, -7.0194e-02,  6.7957e-01,
         4.3002e-01, -9.2223e-01,  7.3037e-01, -1.4577e-01, -7.1286e-01,
        -4.0511e-01, -1.6829e-01, -8.2194e-01,  5.9677e-02,  1.0802e-01,
        -1.3944e-01, -8.4483e-02,  1.1963e-01, -6.0342e-01, -1.3646e-01,
        -2.0323e-01, -4.3106e-02,  8.6185e-02, -9.6659e-02,  3.4307e-02,
        -4.0804e-01, -1.3334e-01, -1.8088e-01, -9.4683e-02,  5.8635e-01,
        -3.4217e-01,  9.1283e-02,  4.1226e-01,  2.8861e-02, -5.1508e-01,
        -1.2082e-01, -1.5805e+00,  1.6071e-01, -5.1493e-01, -1.8585e-01,
        -5.5214e-01,  1.4069e-01,  8.1542e-01,  8.0309e-01,  8.4434e-02,
        -1.2085e-01, -7.0994e-02,  1.7360e-01,  1.9107e-01, -3.8068e-01,
        -1.9188e-01,  6.9382e-01,  3.7637e-02,  1.4737e-01, -1.1562e-01,
         6.4168e-02, -4.5421e-01,  4.8785e-02,  7.9495e-03, -8.5442e-02,
         1.8009e-01,  3.7688e-03,  2.5055e-01,  6.5692e-01, -1.3484e-01,
        -1.0614e+00,  3.0300e-01,  5.7702e-01, -2.1633e-01, -3.8771e-02,
         1.2441e+00,  2.9152e-01, -2.8115e-01]

#model inference
def model_inference(model, dataloader):
    global DEVICE, BOS_TOKEN_LAST
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
            # found = 0
            # for i in range(in_embs.size(1)):
            #     if torch.all(torch.isclose(torch.tensor(SAMPLE), in_embs[0, i, :].cpu())):
            #         found = i
            #         break
            # assert found
            # print("Found at: ", found)
            in_embs, target_embs, target_tokens, var_index_mask_no, lambda_index_mask, app_index_mask, stop_mask, pad_mask, sent_pad_mask = in_embs.to(DEVICE), target_embs.to(DEVICE), target_tokens.to(DEVICE), var_index_mask_no.to(DEVICE), lambda_index_mask.to(DEVICE), app_index_mask.to(DEVICE), stop_mask.to(DEVICE), pad_mask.to(DEVICE), sent_pad_mask.to(DEVICE)
            BOS_TOKEN_LAST = target_embs[0, 0, :].unsqueeze(0).unsqueeze(0)
            seq_syntax = var_index_mask_no.type(torch.bool) + 2*lambda_index_mask.type(torch.bool) + 3*app_index_mask.type(torch.bool) 
            out, out_, classified_class, var_reg = model(target_embs[:, :-1, :], seq_syntax[:, :-1], in_embs, sent_pad_mask) # get_discrete_output(in_embs, model, target_tokens.shape[1])
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
            # print(pr)
            #pr = torch.gather(pr, -1, gt_cls_mask.unsqueeze(-1)).squeeze(-1)
            pr = torch.gather(pr, -1, pr.argmax(-1,keepdim=True)).squeeze(-1)
            
            #Write the written outputs:
            out = out[0].argmax(-1) if len(out.shape) == 3 else out[0]
            outs.append([list(zip(gt_cls_mask.squeeze(0).tolist(), pr.squeeze(0).tolist())), get_out_list(out, classified_class, var_reg), pr.prod(dim=-1).squeeze(0).item()])
            
            beam_size = 12
            x, y, z, p, allprobs_seq = get_discrete_output(in_embs, model, tokenized=True, last=False, max_len=200, unfiltered_class=True, beam_size=beam_size, reference=target)
            if beam_size <= 1: outs.append([y.argmax(-1).squeeze(0).tolist(), get_out_list(x, y, z), p])
            else:
                for jkl in range(len(y)):
                    outs.append([list(zip(y[jkl].tolist(), allprobs_seq[jkl].tolist())) , get_out_list(x[jkl], y[jkl], z[jkl]), p[jkl].item()])
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
            if k>1: break
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


def get_discrete_output(input_embs, model, max_len=20, tokenized=False, last=False, unfiltered_class=False, beam_size=1, reference=None):
    model.eval()
    #bert tokenize and get embeddings
    if not tokenized: 
        tokenized = TOKENIZER(input_embs, return_tensors="pt", padding=True)
        input_embs, _ = get_bert_emb(tokenized)
    input_embs = input_embs.to(DEVICE)
    #model inference
    out, classified_class, var_reg, pr, prs = model(input_embs, max_len=max_len, last=last, unfiltered_class=unfiltered_class, beam_size=beam_size, reference=reference)
    return out, classified_class, var_reg, pr, prs

def get_out_list(out, classified_class, var_reg):
    if len(classified_class.shape) > 1: classified_class = classified_class.argmax(dim=-1).squeeze(0)
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
    # print(out_list)
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

def find_best_out(out_vector, in_vectors):
    #closest euclidean distance:
    vecs = (in_vectors - out_vector.unsqueeze(1))**2
    vecs = vecs.sum(dim=-1)
    best_v_index = vecs.argmin(dim=-1)
    print("least value: ", vecs.min())
    best_vector = in_vectors[list(range(in_vectors.size(0))), best_v_index]
    return best_vector

class ClassifierModel(nn.Module):
    def __init__(self, model, linear):
        super().__init__()
        self.model = model
        self.linear = linear
    
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
        return self.linear(outs), outs, classified_class, var_emb

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




