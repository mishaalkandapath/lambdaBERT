from transformers import BertTokenizerFast, BertModel
import torch
import re
from collections import defaultdict
import random

TOKENIZER = BertTokenizerFast.from_pretrained ("bert-base-multilingual-cased")

BERT_MODEL = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)

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

OPEN_RRB = [-1.6961e+00, -2.0422e+00, -1.0674e+00, -3.2021e+00, -6.9292e-01,
         2.4154e+00,  1.4091e+00, -5.1404e-01,  6.5184e-02, -2.0265e+00,
        -1.7034e-01,  1.8869e+00, -8.0829e-01,  7.5037e-01,  1.6284e+00,
         1.9944e+00,  1.1139e+00,  1.5102e+00,  4.1484e+00,  3.3863e+00,
        -8.7745e-01, -7.4159e-01, -4.5839e+00,  3.7352e+00,  1.5515e+00,
         1.7917e+00,  6.4000e-01,  1.6950e+00, -2.5841e+00,  1.0166e+00,
         2.3752e+00, -7.3937e-02,  6.5777e-01, -1.0625e+00, -1.9552e+00,
        -1.8158e+00, -2.0388e+00,  1.7874e+00, -4.8322e+00,  2.9802e+00,
         1.5244e+00, -2.2772e+00,  4.3051e+00, -2.3716e+00, -1.1754e+00,
         4.4284e+00,  2.8018e+00, -1.7409e+00, -1.2170e+00, -2.6682e-01,
        -5.3496e+00,  1.8810e+00,  7.4366e-01,  3.1942e+00, -2.5183e+00,
        -4.9682e+00, -2.5894e+00, -4.0731e-01, -2.5431e+00,  1.9351e-01,
         3.1433e-02, -1.5575e+00,  2.8980e+00,  7.1151e-01,  1.1598e+00,
         1.0421e+00,  7.2613e-01, -1.9346e+00, -5.0683e+00,  2.0833e+00,
         1.5775e+00,  2.1978e+00, -1.5595e-01,  2.1762e+00,  5.3255e+00,
        -1.8626e+00,  2.3395e+00,  7.3629e-01,  1.9904e+00, -5.3136e-01,
         9.5281e-01,  2.0601e+00, -2.1127e+00, -3.3981e-01, -3.3191e-01,
        -1.0453e+00, -1.8583e+00, -5.2164e-01, -2.0311e+00, -6.3054e-01,
        -5.7385e+00, -1.7198e+00, -3.5721e+00, -6.8094e-01,  2.9738e+00,
        -1.0763e+00, -1.1112e+00, -6.0012e-01,  2.4744e-01, -3.7398e+00,
        -2.3289e+00, -4.8977e+00, -9.3011e-01,  4.1668e-01, -1.7510e-01,
         1.1735e+00, -3.1149e+00, -5.4114e+00,  1.3469e+00, -1.0527e+00,
        -9.8216e-01,  1.1082e-01,  1.0169e+00, -4.3010e-01, -1.7000e+00,
         4.4585e+00,  3.1620e-01,  2.6501e+00, -3.5849e+00,  3.7631e+00,
         1.3025e+00, -6.7341e-01, -5.9563e-01,  1.7213e+00, -3.2804e-01,
        -1.5905e+00,  7.3711e-01, -2.3362e+00,  8.5947e-02,  8.6850e-01,
         6.6069e-01, -1.7379e-01, -2.3886e+00, -4.1177e-01,  5.7395e-01,
         2.5181e-01,  1.2002e+00, -1.6153e+00,  2.0633e+00, -2.1603e+00,
         1.2865e+00,  2.3392e+00,  8.3527e-01, -2.9933e+00,  6.2104e-01,
        -1.5871e+00, -1.5603e+00,  3.3265e-02, -3.0403e+00,  2.9055e-01,
        -2.8870e+00,  1.8793e+00, -2.0055e+00,  1.4061e+00, -1.2149e+00,
        -2.7013e+00,  4.7310e-01,  2.6423e+00, -2.2577e+00,  2.6455e+00,
         3.9157e+00,  1.5465e+00, -8.4097e-01,  1.4340e+00, -1.6180e+00,
        -2.2316e+00, -1.8280e+00,  2.4066e+00,  1.2534e+00,  2.0239e+00,
        -4.1864e+00, -4.5658e+00,  1.5222e+00, -2.3995e+00, -1.6374e+00,
         2.6784e+00, -3.5471e-01,  1.8330e+00, -3.4033e-01, -8.5155e-01,
        -1.3926e+00,  2.3794e+00, -1.8515e+00,  4.7972e-02,  2.6718e+00,
         9.2814e-01,  1.4830e+00, -2.9340e+00,  2.1560e+00,  2.1861e+00,
        -1.2146e+00,  4.2638e-01,  3.6490e-01, -1.0657e+00, -9.0493e-03,
        -4.4904e+00, -8.0832e-02, -1.8616e+00,  2.8086e+00, -3.0018e+00,
         2.1633e+00, -1.2720e+00, -9.8114e-01,  2.6729e+00, -2.2918e+00,
         2.8787e+00,  8.9875e-01, -3.1974e+00,  2.9112e+00, -1.4325e+00,
         8.1388e-01,  3.2112e+00, -1.7384e+00, -3.9465e-01, -2.9474e-01,
        -9.6165e-01,  1.9185e+00, -9.0525e-01,  9.4798e-01, -3.1953e+00,
         7.5623e-01, -4.5319e+00,  5.7622e-01,  3.2002e-01, -4.4746e+00,
         2.9433e+00,  6.9177e-01,  5.3807e-01,  3.3243e+00, -3.0679e+00,
        -2.2305e+00, -4.7261e+00,  3.2123e+00,  1.1429e+00,  2.1933e+00,
         1.3125e+00, -2.0459e-01, -7.8156e-01, -9.5011e-01, -2.4314e+00,
         2.0503e+00, -4.0303e-01,  1.1590e+00,  1.8183e-01,  6.1332e-01,
         3.5937e-01,  2.8917e+00, -4.7690e+00, -3.5002e-01,  9.1190e-01,
        -2.6377e+00, -2.5803e-01,  8.6425e-01, -6.3921e-01, -2.3924e-01,
        -3.9379e+00, -7.1618e-01, -1.6393e+00,  3.1670e+00, -6.6433e-01,
         4.9355e-01,  8.0072e-01,  2.5679e+00, -1.5016e+00, -1.4663e+00,
         9.1624e-01, -1.6826e+00, -1.6722e+00, -1.4309e+00, -4.3222e+00,
        -2.8191e+00,  1.2341e+00, -2.5072e+00, -3.0831e+00, -4.1961e+00,
         8.6770e-01, -6.5037e-02, -1.9207e+00,  9.5848e-01,  4.6156e+00,
         7.3115e-01,  4.3483e-01, -1.9719e+00, -9.4572e-01,  4.4098e+00,
        -9.0579e-01, -3.3964e+00, -2.0920e+00,  1.1643e+00, -1.0710e+00,
        -1.3980e+00,  3.0292e+00,  1.5874e+00, -1.7424e+00,  1.0746e+00,
         2.5482e-02,  4.1966e-01, -1.0426e+00, -1.9230e+00, -7.5293e-01,
         1.3214e+00,  2.6582e+00,  2.4018e-01, -1.7122e+00,  4.1585e+00,
         1.4960e+00, -2.0499e+00,  2.4872e+00, -2.4720e+01, -1.3077e+00,
         2.8969e-01, -1.7807e+00,  5.2185e-02, -1.6921e+00,  1.9651e+00,
        -9.0436e-01, -4.2033e+00, -6.3004e-01,  3.3973e+00, -3.6973e+00,
         2.1583e+00,  1.1718e+00,  2.3586e-01, -1.4293e+00, -1.0927e+00,
        -2.7247e+00, -4.3233e-01,  3.9446e+00, -1.2609e+00, -1.3599e+00,
         2.8010e+00, -2.1461e+00, -4.1305e+00,  7.8284e-02,  1.3296e+00,
         2.2350e+00, -6.4625e+00,  1.5635e+00, -1.6659e+00, -6.8757e-01,
         6.7574e-01,  6.6134e+00,  6.6645e-04,  2.5962e+00,  2.7438e+00,
        -2.2002e+00,  5.0677e+00,  7.2657e-01,  2.3329e+00,  1.6440e+00,
        -5.3143e-01,  2.4088e+00,  2.8364e+00,  3.0615e-01,  2.6672e+00,
         1.7359e+00,  2.7138e-01,  3.2570e-01,  5.5201e-01, -2.7636e+00,
         9.3898e-01, -8.6797e-01, -1.4763e+00,  2.2335e+00,  2.6227e+00,
         3.1539e+00,  1.5660e+00, -1.8300e-01, -6.8947e-01, -3.8640e+00,
         1.0240e+00,  2.2784e+00, -4.7940e+00, -9.3738e-01, -3.0787e+00,
         2.5984e+00,  2.8046e+00, -3.7207e-01, -2.2103e+00,  2.1580e+00,
        -2.3156e+00, -1.1262e+01, -2.5952e+00, -1.4019e+00,  7.5979e-01,
        -2.4221e+00,  6.0770e-01, -3.6796e+00, -4.7907e+00, -2.2612e+00,
        -1.0210e+00,  3.1658e+00, -3.1337e+00,  1.7480e+00, -4.6236e-01,
         2.9617e-01, -2.2566e+00,  7.1449e-01,  7.2416e-01,  1.9374e-01,
         1.4780e+00,  1.1667e+00,  1.8519e-01,  1.5098e+00,  4.2808e+00,
        -6.8178e-01,  1.5543e+00,  3.5067e+00, -3.4621e+00,  3.6763e+00,
         6.6310e-01,  1.5020e+00,  1.2753e+00, -9.6668e-01, -3.0798e+00,
        -3.7162e-02,  9.7457e-01, -1.4209e+00,  5.9346e-01, -2.7569e+00,
        -2.3192e+00,  1.5222e+00,  7.1953e-01, -1.3383e+00,  1.1373e+00,
        -3.9692e+00, -3.2520e-01,  1.3425e+00, -4.4285e+00, -1.5919e+00,
        -2.8518e+00, -4.4901e+00, -5.4101e-01, -2.6228e+00,  7.6148e-01,
        -1.8706e+00, -2.5793e-01,  9.6222e-01,  2.7280e+00,  9.3236e-01,
         2.0571e-01,  1.7552e+00, -2.4023e+00, -4.0125e+00,  1.5290e+00,
         2.3530e-01,  3.5765e+00, -1.4883e+00,  1.8932e+00,  8.6151e-01,
        -2.8605e+00,  1.5433e+00,  2.3539e+00,  1.5515e-01, -9.2036e-01,
         1.3793e+00,  4.2253e-01,  3.8895e+00,  2.4011e+00, -1.8777e+00,
         3.1947e+00,  1.3093e+00, -2.3780e+00, -4.4316e+00,  1.6977e+00,
        -1.7530e+00, -1.5433e+00,  1.6549e+00,  3.3100e+00, -6.3364e-01,
        -1.9421e+00,  6.9846e-01,  5.0258e+00,  1.8339e+00, -2.0303e+00,
         6.8069e-01,  1.1632e+00, -2.0962e+00,  1.3267e+00, -1.7468e+00,
        -2.9267e+00, -1.4303e-01,  2.6801e+00, -2.0688e+00, -1.9683e+00,
        -4.3596e+00, -1.5459e+00,  1.9205e+00,  4.3688e-01,  7.8965e-01,
        -5.2332e-01, -8.3964e-01, -1.3203e-03,  1.6331e+00, -3.0622e+00,
        -9.2503e-01,  6.0332e-01,  6.0411e+00,  5.5061e-01, -1.3670e+00,
         4.0810e-01,  4.3478e-01, -2.7691e+00, -1.7854e+00,  1.8252e+00,
        -7.5156e-01,  8.9125e-01,  1.1144e-02, -4.7463e+00, -3.3891e+00,
         1.9001e+00,  3.9432e-01,  1.7373e+00,  3.8323e+00, -2.8319e+00,
         2.8462e+00, -1.7991e+00, -3.6439e-01,  4.3300e+00,  8.8645e-01,
        -2.2358e-01, -1.2470e+00, -3.4417e+00,  3.9839e+00,  3.6342e-01,
         2.9943e+00,  3.3266e-01, -3.9151e+00,  5.8184e-01, -1.9564e+00,
        -9.0570e-01,  5.3430e-02,  1.6216e+00,  1.7305e+00,  1.9329e+00,
         9.0219e-01, -1.3724e+00, -1.4880e+00, -4.9689e+00,  2.0910e+00,
         1.0341e+00,  2.9361e+00,  3.6064e+00, -1.2875e+00, -5.7857e-01,
         1.9964e+00, -7.3340e-01,  1.7880e+00,  5.2903e-01, -2.1394e+00,
        -1.7060e+00,  8.7629e-01, -1.5791e+00, -7.4643e-01,  1.1092e+00,
        -1.4456e+00,  7.5136e-01,  4.7553e+00, -8.0595e-01, -2.1451e+00,
         1.6992e+00,  2.8645e-01, -1.8405e+00, -3.6413e+00,  8.0756e-01,
         1.3828e+00, -1.0627e+00, -4.7082e-01, -2.1867e-01,  3.8069e+00,
         7.2077e-01,  1.2954e+00,  3.6877e-01, -5.8064e-01, -1.1223e+00,
         1.0599e-01,  1.5754e+00,  2.9650e+00, -2.3612e+00,  1.5218e+00,
         1.9648e+00,  4.3682e+00,  5.7583e-01, -1.3907e+00,  2.0143e+00,
        -4.0548e+00, -2.0013e+00,  2.6197e+00, -1.4793e+00,  1.2565e+00,
         9.5460e-01,  1.7420e+00,  4.1539e+00, -7.3877e-02, -2.8730e+00,
         3.1950e-01,  1.6404e-01, -1.9337e+00, -1.2701e+00,  1.3421e+00,
         1.3087e+00, -1.6353e+00, -1.4192e+00,  2.1709e+00, -2.5546e+00,
        -3.7331e-01,  2.6944e-01, -1.7492e+00, -2.5385e+00,  1.2498e-01,
        -2.0088e+00,  2.6782e+00, -3.7370e+00,  1.9791e+00, -5.1984e-01,
        -2.2545e+00,  7.3026e-01,  2.1362e+00,  4.6137e-01, -7.7869e-02,
         1.0323e+00,  4.5969e+00,  2.5986e+00,  1.4878e-01, -3.8888e-01,
        -1.3985e+00,  3.8710e-01,  1.2227e+00,  8.6056e-01, -3.7328e+00,
        -1.0564e+00,  2.7315e+00,  3.1288e+00, -2.5105e+00, -1.9290e+00,
        -1.8744e+00, -1.4893e+00,  2.9363e+00,  9.1775e-01,  1.1076e-01,
        -2.9509e-01,  2.9963e+00,  1.5671e+00,  3.7346e-01,  3.6370e+00,
         2.0757e+00,  1.3094e+00,  2.9785e-01, -3.6125e+00, -1.9912e+00,
        -2.7642e+00,  1.0376e-01, -6.5000e-01,  1.7430e-01, -3.3447e+00,
        -2.8044e-01, -2.7744e+00,  5.3123e+00,  3.8018e-01,  1.9502e+00,
        -2.4023e+00,  2.4526e+00, -1.3847e+00, -1.4545e+00,  2.3665e-01,
        -8.7737e-01, -3.0753e+00,  1.5392e+00,  1.3352e+00,  2.3572e+00,
         1.5128e+00, -7.4508e-01,  1.7468e+00,  9.9836e-01,  3.6106e+00,
        -2.3211e+00,  1.4355e+00, -1.5977e+00, -5.4448e-01,  1.0320e+00,
        -1.2304e+00,  4.7022e+00,  8.7680e-01,  2.3519e-01,  3.9593e+00,
         5.6014e-01,  9.8942e-01,  4.6026e-01,  2.7092e+00, -1.3953e-01,
        -1.8739e+00,  1.4642e+00,  4.6972e+00, -2.6591e+00, -7.2744e-01,
         3.4609e-01, -6.5104e-01, -4.8958e+00, -8.6633e-03,  1.1517e-01,
        -1.2366e+00, -7.6128e+00,  1.2527e+00, -1.1398e+00,  2.7182e+00,
         5.7240e-01,  1.5866e+00, -8.1452e-01,  5.8519e-01,  1.7552e+00,
        -7.8614e-01, -2.2964e+00, -4.0479e+00, -2.8071e+00, -6.2524e-01,
         3.4324e+00, -3.4033e-01, -2.1575e+00,  3.0227e+00,  7.7441e-01,
        -6.3353e-02,  4.5243e+00, -6.3948e-01, -7.5664e-01, -3.1141e-01,
         2.4622e-01, -8.3775e-02, -4.4048e+00,  3.1531e+00, -1.0543e-01,
        -1.0045e+00,  3.3816e+00,  1.6777e+00, -4.9275e+00,  2.5141e-01,
         3.2209e+00, -4.4022e-01,  2.5940e+00, -1.6425e+00,  1.2809e+00,
        -8.3048e-01, -2.0874e+00, -1.5069e+00, -2.6095e+00,  3.9389e-01,
        -2.0322e+00,  1.3766e+00, -7.2718e-01, -9.8230e-01,  1.7056e+00,
        -2.1116e-01, -3.1722e+00, -1.9813e+00,  1.1828e+00,  3.7889e-01,
        -3.1476e-02,  3.5646e-01,  1.8780e+00,  1.7995e+00, -2.9479e+00,
        -1.4645e+00, -5.6230e-01, -7.6042e-01,  1.9764e-01,  2.2066e+00,
         1.2697e+00, -1.2788e+00, -3.6808e-01]

def create_out_embeddings(sentences, lamda=False):
    #tokenize tha sentence 
    tokenized = TOKENIZER(sentences, 
                          return_tensors="pt", #return torch tensors
                          padding=True, #pad to max length in batch
                          truncation=True) #truncate to max model length
    #search for the tokenized id of λ
    input_ids = tokenized["input_ids"]
    lambda_id = TOKENIZER.convert_tokens_to_ids("λ")
    app_id = TOKENIZER.convert_tokens_to_ids("(")

    if lamda:
        #get all the indices where the lambda token is presen
        lambda_index_mask = (input_ids == lambda_id)
        app_index_mask = (input_ids == app_id)
        var_index_mask = (input_ids == TOKENIZER.convert_tokens_to_ids("np"))
        var_index_mask = var_index_mask | (input_ids == TOKENIZER.convert_tokens_to_ids("##np"))
        var_index_mask = (input_ids == TOKENIZER.convert_tokens_to_ids("s"))
        var_index_mask = var_index_mask | (input_ids == TOKENIZER.convert_tokens_to_ids("##s"))
        var_index_mask = (input_ids == TOKENIZER.convert_tokens_to_ids("pp"))
        var_index_mask = var_index_mask | (input_ids == TOKENIZER.convert_tokens_to_ids("##pp"))
        var_index_mask = (input_ids == TOKENIZER.convert_tokens_to_ids("n"))
        var_index_mask = var_index_mask | (input_ids == TOKENIZER.convert_tokens_to_ids("##n"))
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
    return (tokenized, lambda_index_mask, app_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask) if lamda else (tokenized, pad_mask)

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

def process_bert_lambda(tokenized_sents, lambda_index_mask, app_index_mask, var_index_mask, lambda_norm=True, var_norm=True):
    assert lambda_norm if var_norm else True, "norm_lambda cant be off and norm_var be on"
    # global BIG_VAR_EMBS
    #get the bert embeddings
    var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask = var_index_mask
    embs = get_bert_emb(tokenized_sents)
    #time to mask out the variables
    if var_norm: 
        embs[var_index_mask | var_index_mask_underscore | pad_mask] = torch.zeros_like(embs[0, 0, :]) # also make hte pad embeddings 0
        mask_sort = torch.argsort((var_index_mask | var_index_mask_underscore).to(torch.uint8), stable=True) #move the embeddiungs to the end
        
        # rearrange everything
        embs = torch.gather(embs, -1, mask_sort.unsqueeze(-1).expand(-1, -1, embs.size(-1)))  # all the var names and the underscores have been moveed to the end
        lambda_index_mask = torch.gather(lambda_index_mask, -1, mask_sort)
        var_index_mask_no = torch.gather(var_index_mask_no, -1, mask_sort)
        app_index_mask = torch.gather(app_index_mask, -1, mask_sort)  
        var_index_mask = torch.gather(var_index_mask, -1, mask_sort)
        var_index_mask_underscore = torch.gather(var_index_mask_underscore, -1, mask_sort)
        pad_mask = torch.gather(pad_mask, -1, mask_sort) 
        pad_mask = pad_mask | var_index_mask | var_index_mask_underscore #extend the pad mask
        
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

    return embs, lambda_index_mask, app_index_mask, var_index_mask, var_index_mask_underscore, var_index_mask_no, pad_mask

def create_out_tensor(sentence, lambda_term):
    # group the offsets into a word dictionary
    # for example {(0, 8): [(0, 2), (2, 6), (6, 8)]}

    # create a map from word in lambda term to original sentence as such:
    # {(3, 6): (8, 11)}

    # then for every offset that is not in the map, tokenize it as a character or word as suitable
    # for every offset in lambda term that is in the map, replace w the tokens in the corresponding list as value in the first map

    # making the first map
    # t = TreebankWordTokenizer()
    # words = t.tokenize(sentence)
    words = " ".join(sentence).replace("...}"," ...}").replace("{..","{. .").replace("NP.","NP .").replace("NP—","NP —").replace(",}"," ,}").\
    replace("'re"," 're").replace("'s"," 's").replace("'ve}"," 've}").replace("!}"," !}").replace("?}"," ?}").replace("n't"," n't").\
    replace("'m}"," 'm}").replace("{. ..","{...").replace("{——}","{— —}").replace("{--—}","{- -—}").replace("St.", "St").split()
    
    words = [word[:-1] for i, word in enumerate(words) if i % 2 != 0]

    ind_letters = set()

    for i, word in enumerate(words):
        if "." in word and len(list(set(word))) != 1: words[i] = words[i].replace(".", "")
        if len(word) == 1 and word.isalpha(): ind_letters.add(i)
    
    replacement = "J"
    while replacement in words:
        replacement = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[random.randint(0, 25)]

    tokens = TOKENIZER(" ".join(words), add_special_tokens=True, return_tensors="pt")
    word_mapping = tokens.words()
    word_mapping[0] = -1
    word_mapping[-1] = -1

    # print(words)

    #get the bert embedding:
    representations = get_bert_emb(tokens)

    #ptb tokenize the lambda term 
    lambda_term_list = []
    acc = ""

    weird_dots = re.findall(r"<w_\d+>", lambda_term)
    for i, dot in enumerate(weird_dots):
        lambda_term = lambda_term.replace(dot, f"..._{i+1000}")

    for char in lambda_term:
        if char in "( )":
            if acc != "":
                lambda_term_list.append(acc)
                acc = ""
            if char != " ": lambda_term_list.append(char)
        else: 
            acc += char
    replace_copy = lambda_term_list.copy()
    term_to_word_index = {}
    # print(lambda_term_list)

    #first compile a list of variable positions:
    var_logs = []
    lambda_pattern = re.compile(r"λ\w+_\d+\.")
    for i, element in enumerate(lambda_term_list):
        if lambda_pattern.match(element)is not None:
            var_logs.append(element[1:-1])

    for w, word in enumerate(words):
        #find if this constitutes an entity in the lambda term
        #entities are of the form words_dddd
        if word == "(": word = "LRB"
        elif word == ")": word = "RRB"
        elif word == "[": word = "LSB"
        elif word == "]": word = "RSB"
        elif word == "{": word = "LCB"
        elif word == "}": word = "RCB"
        # word = word
        if "λ" in word: word = word.replace("λ", "")
        if word not in lambda_term: continue
        #traverse the lambda_term 
        min_indx, min_no = 500000, 500000
        for i, term in enumerate(lambda_term_list):
            if re.findall(r"_\d+", term) and term[: term.rfind(re.findall(r"_\d+", term)[0])] == word and term not in var_logs and "_" in term:
                no = int(term[term.rfind(re.findall(r"_\d+", term)[0])+1:])
                if no < min_no:
                    min_no = no
                    min_indx = i
        if min_indx == 500000: continue
        #replace with something random
        lambda_term_list[min_indx] = replacement*len(word)
        term_to_word_index[min_indx] = w

    lambda_term_list = replace_copy

    lambda_term_embedding = []
    lambda_term_tokens = []
    var_mask, lambda_mask, app_mask = [], [], []

 
    var_pattern1 = re.compile(r"S_\d+")
    var_pattern2 = re.compile(r"NP_\d+")
    var_pattern3 = re.compile(r"N_\d+")
    var_pattern4 = re.compile(r"PP_\d+")
    entity_pattern = re.compile(r".*_\d+")

    var_logs = []
    # print(lambda_term_list)
    # print(term_to_word_index)
    # print(words)
    for i, element in enumerate(lambda_term_list):
        if element == "(" or element == ")":
            lambda_term_embedding.append("loo" if element == ")" else OPEN_RRB)
            app_mask.append(1)
            lambda_mask.append(0)
            var_mask.append(0)
            lambda_term_tokens.append(-1)
        elif lambda_pattern.match(element)is not None:
            lambda_term_embedding.append(LAMBDA)
            var_mask.append(0)
            lambda_mask.append(1)
            app_mask.append(0)
            lambda_term_tokens.append(-1)

            lambda_term_embedding.append(torch.rand((768,)))
            var_logs.append(element[1:-1])
            lambda_mask.append(0)
            var_mask.append(len(var_logs))
            app_mask.append(0)
            lambda_term_tokens.append(-1)
        elif (var_pattern1.match(element) is not None or var_pattern2.match(element) is not None or var_pattern3.match(element) is not None or var_pattern4.match(element) is not None) and element in var_logs:
            assert element in var_logs, "Variable not found in lambda term?? "+lambda_term + " " + element
            lambda_term_embedding.append(torch.rand((768,)).tolist())
            lambda_mask.append(0)
            var_mask.append(var_logs.index(element)+1)
            app_mask.append(0)
            lambda_term_tokens.append(-1)
        else:
            assert entity_pattern.match(element), "Invalid lambda term ?? " + element
            counts = torch.count_nonzero(torch.tensor(word_mapping) == term_to_word_index[i])
            lambda_term_embedding.extend(representations.squeeze(0)[torch.tensor(word_mapping) == term_to_word_index[i]])
            lambda_term_tokens.extend(tokens.input_ids[0][torch.tensor(word_mapping) == term_to_word_index[i]].tolist())

            lambda_mask.extend([0]*counts)
            var_mask.extend([0]*counts)
            app_mask.extend([0]*counts)

        assert len(var_mask) == len(lambda_mask) == len(lambda_term_embedding) == len(app_mask) == len(lambda_term_tokens), f"{len(var_mask)} {len(lambda_mask)} {len(app_mask)} {len(lambda_term_embedding)} {len(lambda_term_tokens)}"
    return representations, torch.tensor(lambda_term_embedding), lambda_term_tokens, var_mask, lambda_mask, app_mask


if __name__ == "__main__":
    import pandas as pd
    import os
    from tqdm import tqdm
    df = pd.read_csv("data/input_sentences.csv", header=None)
    sentences = len(df)     

    for i in tqdm(range(sentences)):
        # error here: i+9 + 56+75+477+25, i+9 + 56+75+477+26+128, i+9 + 56+75+477+26+129+278, i+9 + 56+75+477+26+129+279+111+724, i+9 + 56+75+477+26+129+279+111+725+482, i+9 + 56+75+477+26+129+279+111+725+483+6606+3757, i+9 + 56+75+477+26+129+279+111+725+483+6606+3757+157, i+9 + 56+75+477+26+129+279+111+725+483+6606+3758+158+5809, i+9 + 56+75+477+26+129+279+111+725+483+6606+3758+158+5809+4, i+9 + 56+75+477+26+129+279+111+725+483+6606+3758+158+5809+4+2113, i+9 + 56+75+477+26+129+279+111+725+483+6606+3758+158+5810+5+2114 + 2626,i+9 + 56+75+477+26+129+279+111+725+483+6606+3758+158+5810+5+2114 + 2626 + 1396,i+9 + 56+75+477+26+129+279+111+725+483+6606+3758+158+5810+5+2114+2627+1398+200, i+9 + 56+75+477+26+129+279+111+725+483+6606+3758+158+5810+5+2114+2627+1398+200+3440, +4453, +28,+10447, +5700
        gen_sent = eval(df.iloc[i, 1])

        path = df.iloc[i, 2]
        path = "/w/150/lambda_squad/lambdaBERT/data/" + path[len("lambdaBERT/data/"):]
        with open(path, 'r') as f:
            lambda_terms = f.readlines()[0].strip()
        lambda_terms = lambda_terms.replace(")", "")
        sent_emb, target_emb, target_tokens, var_mask, lambda_mask, app_mask = create_out_tensor(gen_sent, lambda_terms)
        # print(target_emb)
        # print(var_mask)
        # print(lambda_mask)
        # print(f"/w/150/lambda_squad/{df.iloc[i, 2][:-4]}.pt")
        torch.save((sent_emb, target_emb, target_tokens, var_mask, lambda_mask, app_mask), f"/w/150/lambda_squad/{df.iloc[i, 2][:-4]}.pt")
        
    # for i in [67254, 57102, 40593, 43650]:
    #     gen_sent = eval(df.iloc[i, 1])
    #     words = " ".join(gen_sent).replace("...}"," ...}").replace("{..","{. .").replace("NP.","NP .").replace("NP—","NP —").replace(",}"," ,}").\
    # replace("'re"," 're").replace("'s"," 's").replace("'ve}"," 've}").replace("!}"," !}").replace("?}"," ?}").replace("n't"," n't").\
    # replace("'m}"," 'm}").replace("{. ..","{...").replace("{——}","{— —}").replace("{--—}","{- -—}").replace("St.", "St").split()
    #     words = [word[:-1] for i, word in enumerate(words) if i % 2 != 0]

    #     tokens = TOKENIZER(" ".join(words), add_special_tokens=True, return_tensors="pt")
    #     representations = get_bert_emb(tokens)

    #     path = df.iloc[i, 2]
    #     sent_emb = representations
    #     target_emb = representations[0, 1, :].reshape((1, 1, 768))
    #     var_mask = [0]
    #     lambda_mask = [0]
    #     app_mask = [0]
    #     # target_emb = representations[]
    #     torch.save((sent_emb, target_emb, var_mask, lambda_mask, app_mask), f"/w/150/lambda_squad/{df.iloc[i, 2][:-4]}.pt")






        
        
        
        
        
    