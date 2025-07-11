import matplotlib.pyplot as plt
from models import *
import dataloader
import tqdm, csv, random, pandas as pd
from tokenization import get_bert_emb, TOKENIZER, LAMBDA, OPEN_RRB, BERT_MODEL, preprocess_sent
from dataloader import SEP_TOKEN, BOS_TOKEN, BOS_TOKEN_LAST
from inference import teacher_forcing

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
from copy import deepcopy

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
def plot_confusion_matrix(confusion_matrix, split="train", plot_name="train"):
    # Label mapping
    label_map = {
        0: "Word",
        1: "Variable",
        2: "Lambda",
        3: "Application"
    }
    
    # Normalize confusion matrix
    normalized_matrix = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True)
    
    # Create figure with a larger size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(
        normalized_matrix,
        annot=True,  # Show numbers in cells
        fmt='.2%',   # Format as percentage with 2 decimal places
        cmap='YlGnBu',  # Different colormap that's easier to read
        xticklabels=list(label_map.values()),
        yticklabels=list(label_map.values()),
        square=True,  # Make cells square
        cbar_kws={'label': 'Proportion'}
    )
    
    # Customize the plot
    plt.title(f'Confusion Matrix ({split.capitalize()})', pad=20)
    plt.xlabel('Predicted Label', labelpad=10)
    plt.ylabel('True Label', labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save and clear
    plt.savefig(f"confusion_matrix_{plot_name}.png", dpi=300, bbox_inches='tight')
    plt.clf()

def easy_free_variable_counts(input_sents, model):
    model.eval()

    tokenized = TOKENIZER[os.environ["BERT_TYPE"]](input_sents, return_tensors="pt", padding=True)
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
    num_vars_a = sum([1 for t in str1 if re.match(r"x\d+", t)])
    num_vars_b = sum([1 for t in str2 if re.match(r"specvar\d+", t)])
    rename = lambda x, t: "specvar"+ str(int(x[1:]) + t)
    new_str2 = [t if not re.match(r"x\d+", t) else rename(t, num_vars_b) for t in str1] # offset so that no two variables are shared acorss str1 and str2

    var_name_counter = num_vars_a + num_vars_b

    str2_var_pointers = {}
    new_new_str2 = []
    for i, t in enumerate(new_str2):
        if re.match(r"specvar\d+", t):
            if t in str2_var_pointers: 
                new_new_str2.append(str2_var_pointers[t])
            else:
                str2_var_pointers[t] = [t]
                new_new_str2.append(str2_var_pointers[t])
        else:
            new_new_str2.append(t)
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
                and (re.match(r"specvar\d+", str1[t1-1]) and re.match(r"specvar\d+", access_b(t2-1))):
                c = distances[t1 - 1][t2 - 1] # no need to change anything here -- simple variable renaming
                #switch pointers around in b 
                #make new var name
                new_var_name = f"x{var_name_counter}"
                var_name_counter += 1

                str2_var_pointers_copy = deepcopy(str2_var_pointers)
                str1_copy = deepcopy(str1)
                new_new_str2_copy = deepcopy(new_new_str2)

                old_access = access_b(t2-1)
                # print(str2_var_pointers, new_var_name)
                str2_var_pointers[old_access][0] = new_var_name
                str2_var_pointers[new_var_name] = str2_var_pointers[old_access]

                s1 = " ".join(str1)
                s1 = s1.replace(str1[t1-1], new_var_name)
                str1 = s1.split()
                
                a = distances[t1][t2 - 1] + 1
                b = distances[t1 - 1][t2] + 1

                if (a <= b and a <= c):
                    distances[t1][t2] = a
                    #reset renaming
                    str2_var_pointers = str2_var_pointers_copy
                    str1 = str1_copy
                    new_new_str2 = new_new_str2_copy
                    var_name_counter -= 1
                elif (b <= a and b <= c):
                    str2_var_pointers = str2_var_pointers_copy
                    str1 = str1_copy
                    new_new_str2 = new_new_str2_copy
                    var_name_counter -= 1
                    distances[t1][t2] = b
                else:
                    distances[t1][t2] = c
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

def levenshtein_distance_for_lists(list2, list1):
    """
    Calculate the Levenshtein distance between two lists of strings.
    The distance is the minimum number of single-element edits (insertions, 
    deletions, or substitutions) required to change one list into the other.
    
    Args:
        list1: The first list of strings
        list2: The second list of strings
        
    Returns:
        The Levenshtein distance between the two lists
    """
    #initial renaming:
    rename = lambda x: "specvar"+ str(int(x[1:]) - 1)
    
    def split_string(text, pattern=r"(x\d+)(.*)"):
        match = re.match(pattern, text)
        if match:
            return match.groups()
        else:
            return None
    
    new_list1 = []
    for t in list1:
        if re.match(r"x\d+", t):
            renaming = split_string(t)
            if len(renaming[-1]) == 0: renaming = renaming[:-1]
            new_list1.extend(renaming)
        else:
            new_list1.append(t)
    list1 = new_list1

    #similarly for list2
    new_list2 = []
    for t in list2:
        if re.match(r"specvar\d+", t):
            renaming = split_string(t, pattern=r"(specvar\d+)(.*)")
            if len(renaming[-1]) == 0: renaming = renaming[:-1]
            new_list2.extend(renaming)
        else:
            new_list2.append(t)

    list2 = new_list2

    list1 = [t if not re.match(r"x\d+", t) else rename(t) for t in list1] # offset so that no two variables are shared acorss str1 and str2

    # Create a matrix to store the distances
    # Size is (len(list1) + 1) x (len(list2) + 1)
    dp = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]
    
    # Initialize the first row and column
    for i in range(len(list1) + 1):
        dp[i][0] = i
    for j in range(len(list2) + 1):
        dp[0][j] = j
        
    # Fill the matrix
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            # If the elements are the same, no edit is needed
            if list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                # Take the minimum of the three possible operations:
                # 1. Insert (dp[i][j-1] + 1)
                # 2. Delete (dp[i-1][j] + 1)
                # 3. Substitute (dp[i-1][j-1] + 1)
                dp[i][j] = min(dp[i][j-1] + 1,      # Insert
                               dp[i-1][j] + 1,      # Delete
                               dp[i-1][j-1] + 1)    # Substitute
                
    # The final cell contains the Levenshtein distance
    return dp[len(list1)][len(list2)]

def levenshtein_with_mismatch_pairs(list2, list1):
    """
    Calculate the Levenshtein distance between two lists and return
    the pairs of elements that caused non-zero cost edits.
    
    Args:
        list1: The first list of strings
        list2: The second list of strings
        
    Returns:
        A tuple containing:
        - The Levenshtein distance between the two lists
        - A list of mismatch pairs, where each pair is either:
          - (element1, None) for deletions
          - (None, element2) for insertions
          - (element1, element2) for substitutions
    """
    #initial renaming:
    rename = lambda x: "specvar"+ str(int(x[1:]) - 1)
    
    def split_string(text, pattern=r"(x\d+)(.*)"):
        match = re.match(pattern, text)
        if match:
            return match.groups()
        else:
            return None
    
    new_list2 = []
    for t in list2:
        if re.match(r"x\d+", t):
            renaming = split_string(t)
            if len(renaming[-1]) == 0: renaming = renaming[:-1]
            new_list2.extend(renaming)
        else:
            new_list2.append(t)
    list2 = new_list2

    #similarly for list2
    new_list1 = []
    for t in list1:
        if re.match(r"specvar\d+", t):
            renaming = split_string(t, pattern=r"(specvar\d+)(.*)")
            if len(renaming[-1]) == 0: renaming = renaming[:-1]
            new_list1.extend(renaming)
        else:
            new_list1.append(t)

    list1 = new_list1

    list2 = [t if not re.match(r"x\d+", t) else rename(t) for t in list2] # offset so that no two variables are shared acorss str1 and str2


    # Create a matrix to store the distances
    dp = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]
    
    # Initialize the first row and column
    for i in range(len(list1) + 1):
        dp[i][0] = i
    for j in range(len(list2) + 1):
        dp[0][j] = j
        
    # Fill the matrix
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1] + 1,      # Insert
                               dp[i-1][j] + 1,      # Delete
                               dp[i-1][j-1] + 1)    # Substitute
    
    # Backtrack to find the mismatch pairs
    i, j = len(list1), len(list2)
    mismatch_pairs = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and list1[i-1] == list2[j-1]:
            # Match - move diagonally (no mismatch to report)
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            # Insertion - move left
            mismatch_pairs.append((None, list2[j-1]))
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            # Deletion - move up
            mismatch_pairs.append((list1[i-1], None))
            i -= 1
        else:
            # Substitution - move diagonally
            mismatch_pairs.append((list1[i-1], list2[j-1]))
            i -= 1
            j -= 1
    
    # Reverse the pairs to get them in the correct order (from start to end)
    mismatch_pairs.reverse()
    
    return dp[len(list1)][len(list2)], mismatch_pairs

def compute_confusion_matrix(model, dataloader, last=False):
    global DEVICE
    model.eval()
    confusion_matrix = torch.zeros(4, 4)
    average_loss = 0
    count = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(dataloader))
        for _, batch in enumerate(dataloader):
            _, loss, out, classified_class, var_reg, gt_cls_mask, in_embs, in_tokens, _ = teacher_forcing(model, batch)

            average_loss += loss.item()
            count += 1
            pbar.set_description(f"Loss: {loss.item()/count}")
            pbar.update(1)

            classified_class = classified_class.argmax(dim=-1).squeeze(0)
            gt_cls_mask = gt_cls_mask.squeeze(0)
            for i in range(4):
                for j in range(4):
                    confusion_matrix[i, j] += ((classified_class == j) & (gt_cls_mask == i)).sum().detach().cpu()
    return confusion_matrix

def find_arity(lambda_term):
    # Regular expression to match function applications
    pattern = re.compile(r'\(([a-zA-Z_]+[0-9]*)\s([^()]+)\)')
    
    # Dictionary to store function arities
    arities = {}
    
    # Find all matches in the lambda term
    matches = pattern.findall(lambda_term)
    
    for match in matches:
        function_name = match[0]
        arguments = match[1].strip().split()
        
        # Count the number of arguments
        arity = len(arguments)
        
        # Update the arity in the dictionary
        if function_name in arities:
            if arity > arities[function_name]:
                arities[function_name] = arity
        else:
            arities[function_name] = arity
    return list(arities.values())

def align_lambda_terms(lambda_str, true_str):
    pass

def eval_parses(gold_parse, parse):
    pass

if __name__ == "__main__":
    import os
    import Levenshtein
    # #arguments
    l1_parser = argparse.ArgumentParser()
    l1_parser.add_argument("--confusion_matrix", action="store_true")
    l1_parser.add_argument("--levenshtein", action="store_true")
    l1_parser.add_argument("--csv", type=str, default="all_lev.csv")
    l1_parser.add_argument("--split", type=str, default="train")
    l1_parser.add_argument("--arity", action="store_true")
    l1_parser.add_argument("--model_path", type=str, required=False)
    l1_parser.add_argument("--last", action="store_true")
    l1_parser.add_argument("--custom", action="store_true")
    l1_parser.add_argument("--cpu", action="store_true")
    l1_parser.add_argument("--name", default="train")
    l1_parser.add_argument("--data_path", default="")
    l1_parser.add_argument("--rem_spec_indices", action="store_true")

    args = l1_parser.parse_args()

    if args.confusion_matrix:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        #load model
        model = TransformerDecoderStack(4, 384, 8, 3072, custom=args.custom)
        checkpoint = torch.load(args.model_path)
        model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        model_weights.update({k: torch.zeros_like(v) for k, v in model.state_dict().items() if k not in model_weights})
        model.load_state_dict(model_weights)

        # model = ShuffledTransformerStack.load_from_checkpoint(args.model_path, model).model
        DEVICE = torch.device("cpu") if args.cpu else torch.device("cuda")
        model = model.to(DEVICE)

        # --LOAD DATA
        dataloader, valid_dataloader, test_dataloader = dataloader.data_init(1, last=args.last, inference=True, data_path=args.data_path, rem_spec_sentences=args.rem_spec_indices)


        #-- CONFUSION MATRIX --
        # confusion_matrix = compute_confusion_matrix(model, dataloader)
        # plot_confusion_matrix(confusion_matrix, "test", plot_name=args.name+"_train")

        # confusion_matrix = compute_confusion_matrix(model, valid_dataloader)
        # plot_confusion_matrix(confusion_matrix, "valid", plot_name=args.name+"_valid")

        confusion_matrix = compute_confusion_matrix(model, test_dataloader)
        plot_confusion_matrix(confusion_matrix, plot_name=args.name+"_test")
    elif args.levenshtein:
    
        #open csv file
        great_sentence_pairs = []
        with open(args.csv, "r") as f:
            reader = csv.reader(f)
            lev_scores = []
            lev_scores_normed = []
            lev_scores_modified = []
            lev_scores_modified_normed = []

            lev_types = {"missing constant": 0, "extra constant": 0, "mismatched constant": 0, "constant -> lambda": 0, "constant -> app":0, "constant -> var":0, \
                         "missing variable": 0, "extra variable": 0, "mismatched variable": 0, "variable -> lambda": 0, "variable -> app":0, "variable -> constant":0, \
                            "missing lambda": 0, "extra lambda": 0, "lambda -> app":0, "lambda -> constant":0, "lambda -> var":0, \
                                "missing application": 0, "extra application": 0, "application -> lambda":0, "application -> constant":0, "application -> var":0}
            lev_types_all = {"missing constant": 0, "extra constant": 0, "mismatched constant": 0, "constant -> lambda": 0, "constant -> app":0, "constant -> var":0, \
                         "missing variable": 0, "extra variable": 0, "mismatched variable": 0, "variable -> lambda": 0, "variable -> app":0, "variable -> constant":0, \
                            "missing lambda": 0, "extra lambda": 0, "lambda -> app":0, "lambda -> constant":0, "lambda -> var":0, \
                                "missing application": 0, "extra application": 0, "application -> lambda":0, "application -> constant":0, "application -> var":0}

            for row in tqdm.tqdm(reader, total=16363):
                lev_score, lev_seq = levenshtein_with_mismatch_pairs(row[0].split(), row[1].split())#levenstein_lambda_term(row[0].split(), row[1].split())
                
                if lev_score <= 7:
                    for s in lev_seq:
                        if s[0] is None:
                            if s[1] == "λ":
                                lev_types["missing lambda"] += 1
                            elif s[1] == "(":
                                lev_types["missing application"] += 1
                            elif re.match(r"specvar\d+", s[1]): 
                                lev_types["missing variable"] += 1
                            else:
                                lev_types["missing constant"] += 1
                        elif s[1] is None:
                            if s[0] == "λ":
                                lev_types["extra lambda"] += 1
                            elif s[0] == "(":
                                lev_types["extra application"] += 1
                            elif re.match(r"specvar\d+", s[0]): 
                                lev_types["extra variable"] += 1
                            else:
                                lev_types["extra constant"] += 1
                        elif s[0] == "λ":
                            if s[1] == "(":
                                lev_types["lambda -> app"] += 1
                            elif re.match(r"specvar\d+", s[1]):
                                lev_types["lambda -> var"] += 1
                            else:
                                lev_types["lambda -> constant"] += 1
                        elif s[0] == "(":
                            if s[1] == "λ":
                                lev_types["application -> lambda"] += 1
                            elif re.match(r"specvar\d+", s[1]):
                                lev_types["application -> var"] += 1
                            else:
                                lev_types["application -> constant"] += 1
                        elif re.match(r"specvar\d+", s[0]):
                            if s[1] == "λ":
                                lev_types["variable -> lambda"] += 1
                            elif s[1] == "(":
                                lev_types["variable -> app"] += 1
                            elif re.match(r"specvar\d+", s[1]):
                                lev_types["mismatched variable"] += 1
                            else:
                                lev_types["variable -> constant"] += 1
                        else:
                            if s[1] == "λ":
                                lev_types["constant -> lambda"] += 1
                            elif s[1] == "(":
                                lev_types["constant -> app"] += 1
                            elif re.match(r"specvar\d+", s[1]):
                                lev_types["constant -> var"] += 1
                            else:
                                lev_types["mismatched constant"] += 1
                

                # print("True levenshtein distance: ", "Normalized true lev dist: ", )
                # print("Modified levenhtein distance: ", "Normalized modified lev dist: ", )
                lev_scores_modified.append(lev_score)
                lev_scores_modified_normed.append(lev_score/len(row[0].split()))
                t_lev_scores = Levenshtein.distance(row[0], row[1])
                lev_scores.append(t_lev_scores)
                lev_scores_normed.append(t_lev_scores/len(row[0]))

                if lev_score <= 0:
                    great_sentence_pairs.append((row[0], row[1], lev_score, t_lev_scores, lev_scores_modified_normed[-1], lev_scores_normed[-1]))
        #plot histograms on separate plots
        # create plot with subplots
        # print(len(lev_scores), len(lev_scores_normed))
        
        # Create the figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Data arrays to store bin information
        all_bin_edges = []
        all_bin_counts = []
        all_dataset_names = ["True Levenshtein Distance", 
                            "Normalized True Levenshtein Distance",
                            "Modified Levenshtein Distance", 
                            "Normalized Modified Levenshtein Distance"]
        all_datasets = [lev_scores, lev_scores_normed, lev_scores_modified, lev_scores_modified_normed]

        # Plot histograms using Seaborn and capture bin information
        for i, (data, ax, name) in enumerate(zip(all_datasets, axes, all_dataset_names)):
            # Create the Seaborn histogram
            hist = sns.histplot(data, bins=30, kde=False, ax=ax, label=name)
            
            # Extract the actual bin edges and counts from the histogram
            patch_list = hist.patches
            bin_counts = [patch.get_height() for patch in patch_list]
            bin_edges = [patch.get_x() for patch in patch_list]
            bin_edges.append(patch_list[-1].get_x() + patch_list[-1].get_width())
            
            all_bin_edges.append(bin_edges)
            all_bin_counts.append(bin_counts)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set x-ticks for the last subplot
            if i == 3:
                ax.set_xticks(np.arange(0, 5, 0.5))
            
            # Improve aesthetics
            sns.despine(ax=ax)  # Remove top and right spines

        # Add a main title
        fig.suptitle(f"{args.split} set Levenshtein Distances", fontsize=16)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
        plt.savefig(f"{args.name}_{args.split}_lev.png", dpi=300)
        plt.show()

        # Print bin information for each dataset
        for name, bin_edges, counts in zip(all_dataset_names, all_bin_edges, all_bin_counts):
            print(f"\nBin information for {name}:")
            print("Bin Edges | Count")
            print("-" * 30)
            
            for i in range(len(counts)):
                # Format to show bin ranges (left edge to right edge)
                bin_range = f"{bin_edges[i]:.3f} - {bin_edges[i+1]:.3f}"
                print(f"{bin_range:15} | {int(counts[i])}")

        #print average:
        print("True Levenshtein Distance: ", np.mean(lev_scores))
        print("Normalized True Levenshtein Distance: ", np.mean(lev_scores_normed))
        print("Modified Levenshtein Distance: ", np.mean(lev_scores_modified))
        print("Normalized Modified Levenshtein Distance: ", np.mean(lev_scores_modified_normed))

        #write great sentences
        with open(f"{args.name}_{args.split}_great_sentence_pairs.csv", "w") as f:
            writer = csv.writer(f)
            for row in great_sentence_pairs:
                writer.writerow(row)

        # types of edits:
        # Convert dictionary to DataFrame for Seaborn
        df = pd.DataFrame(list(lev_types.items()), columns=['Type', 'Frequency'])

        # Group the types by their prefix for coloring
        df['Category'] = df['Type'].apply(lambda x: x.split()[0] if '->' not in x else x.split(' -> ')[0])

        # Sort by category and frequency for better visualization
        df = df.sort_values(['Category', 'Frequency'], ascending=[True, False])

        # Create the plot
        plt.figure(figsize=(16, 10))
        ax = sns.barplot(x='Type', y='Frequency', hue='Category', data=df, palette='viridis')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(pad=3)

        # Add title and labels
        plt.title('Frequency of Levenshtein Types', fontsize=16)
        plt.xlabel('Type', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        # Add a legend
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Show the plot
        plt.savefig(f'{args.name}_{args.split}_levenshtein_types.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print the frequency for each type
        print("Levenshtein Type Frequencies:")
        for type_name, frequency in lev_types.items():
            print(f"{type_name}: {frequency}")

        #get the list of edit sequences:
        lev_scores_modified = sorted(lev_scores_modified)
        lev_scores_modified_normed = sorted(lev_scores_modified_normed)
        print("34 percent of sequences have an edit distance <= ", lev_scores_modified[int(0.34*len(lev_scores_modified))])
        print("44 percent of sequences have an edit distance <= ", lev_scores_modified[int(0.44*len(lev_scores_modified))])
        print("53 percent of sequences have an edit distance <= ", lev_scores_modified[int(0.53*len(lev_scores_modified))])
        print("34 percent of sequences have an edit ratio <= ", lev_scores_modified_normed[int(0.34*len(lev_scores_modified_normed))])
        print("44 percent of sequences have an edit ratio <= ", lev_scores_modified_normed[int(0.44*len(lev_scores_modified_normed))])
        print("53 percent of sequences have an edit ratio <= ", lev_scores_modified_normed[int(0.53*len(lev_scores_modified_normed))])
        
        
    elif args.arity:
        #load all the data:
        DATA_PATH = "/w/150/lambda_squad/lambdaBERT/data/"
        in_file, lambda_terms = DATA_PATH + 'input_sentences.csv', DATA_PATH + 'lambda_terms/'
        in_sentences = pd.read_csv(in_file)

        #for each line in file:
        ars = []
        for i, row in tqdm.tqdm(in_sentences.iterrows(), total=in_sentences.shape[0]):
            path = in_sentences.iloc[i, 2]
            #get the lambda term
            path = DATA_PATH + path[len("lambdaBERT/data/"):]
            with open(path, 'r') as f:
                lambda_terms = f.readlines()
            #get the arities
            for i, lambda_term in enumerate(lambda_terms):
                arities = find_arity(lambda_term.strip())
                ars.extend(arities)

        #plot histogram of arities;
        plt.hist(ars, bins=30)
        plt.xlabel("Arity")
        plt.ylabel("Count")
        plt.title("Distribution of Function Arity in Lambda Terms")
        plt.grid(True, alpha=0.3)
        plt.savefig("arity_dist.png")

    # t1 = "( What ( λ x1 ( ( do plants ( λ x2 ( ( ( ( in ( their environment ( λ x3 ( λ x4 ( ( do x3 x4 x1 x2".split()
    # t2 = "( What ( λ specvar0 ( ( do plants ( λ specvar1 ( ( ( ( in ( their environment ( λ specvar2 ( λ specvar3 ( ( do specvar2 specvar3 specvar0 specvar1?".split()

    # levenshtein_distance_for_lists(t1, t2)

