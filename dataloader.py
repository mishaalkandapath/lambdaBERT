import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

import random
import pandas as pd

from tokenization import TOKENIZER, BERT_MODEL, create_out_tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

#create a directory where the key is a csv. each row has first column as the raw text sentence, and the second col being the 
# path to the file that stores all its lambda terms

DATA_PATH = "/w/150/lambda_squad/lambdaBERT/data/"

SEP_TOKEN = [[[ 7.6267e-01,  8.9944e-03, -3.6583e-01,  1.9516e-01, -4.8590e-01,
          -2.4512e-01,  5.0226e-01, -4.1119e-01,  4.1043e-01, -9.3080e-02,
           2.3451e-01,  6.1152e-02,  5.1634e-01, -1.2909e-01, -5.4241e-01,
          -6.7891e-01,  1.6271e-01, -3.3280e-01,  2.9146e-01, -1.7934e-03,
          -1.8548e-01,  2.7439e-01,  3.2378e-01, -3.9362e-01,  3.9103e-02,
          -4.8627e-01, -4.8901e-01,  1.3215e-01, -1.6837e-01, -8.3485e-01,
          -1.1902e-01, -3.7513e-01, -4.1434e-01,  6.5679e-01,  1.6730e-01,
           1.0167e-01,  2.7230e-01, -1.8922e-01, -5.6889e-01, -1.6211e-01,
          -3.0427e-01,  2.1588e-01, -7.1898e-02,  2.5038e-01,  5.4332e-02,
          -2.7465e-02,  6.9104e-01,  6.2793e-01,  4.2105e-02,  1.0751e+00,
           4.9342e-01,  3.7197e-01, -1.7328e-01,  4.6306e-01,  8.5680e-02,
           3.0679e-01,  2.1996e-01, -1.3788e-01,  1.8176e-01,  7.9807e-01,
           1.4034e-01,  2.8091e-01, -1.3855e-01, -4.7647e-01,  6.6795e-01,
          -6.7147e-03, -4.6838e-02, -3.2756e-01, -3.2389e-01, -1.9392e-01,
          -5.9283e-01, -8.2135e-01,  5.6446e-01,  3.4074e-01,  7.0208e-02,
           7.2009e-01, -3.9942e-01,  5.4239e-01, -4.1528e-01, -1.1378e-01,
           2.4622e-01,  6.1872e-03,  1.6288e-01,  1.3336e-01, -8.3746e-02,
          -9.4092e-02, -8.6415e-01, -1.0402e-01, -6.7240e-01, -4.0488e-01,
           3.5545e-01,  5.2807e-01, -1.5584e-01,  3.0006e-01, -1.5826e-01,
           2.2644e-01, -2.9865e-01,  1.5775e-01,  4.6667e-02,  3.2529e-01,
           4.1341e-01, -1.6814e-01,  5.7357e-02,  8.8873e-01, -1.2910e-01,
          -1.9431e-01,  6.2281e-01,  2.2408e-01,  3.8858e-01,  9.1565e-01,
           9.2464e-01, -2.9152e-02,  6.7837e-01,  2.1678e-01, -1.3357e-01,
          -2.5276e-01,  1.1788e-01, -3.1780e-02,  2.9165e-01, -2.0494e-01,
          -4.0776e-01, -4.2348e-01,  3.4800e-01,  2.8637e+00, -3.2686e-01,
          -5.0334e-02,  2.6012e-01, -6.1795e-01,  2.2880e-01, -2.4557e-01,
          -1.9044e-01,  1.3346e-01,  2.7117e-01,  4.7136e-01, -3.4187e-01,
           7.4806e-01, -1.8126e-01,  1.0366e-01, -7.0490e-01,  1.2994e-01,
          -9.5094e-03,  2.4216e-01,  3.6581e-01, -1.0569e+00,  3.2195e-04,
           5.5790e-01,  8.3230e-01,  3.4033e-01,  1.0215e+00, -6.3070e-01,
           9.3840e-01, -5.0309e-01, -3.6165e-02, -1.4434e-01, -5.6641e-01,
          -2.2899e-01, -1.8544e-01, -1.8859e-03,  4.4514e-01,  7.4233e-01,
           2.9224e-01,  3.2790e-01,  6.6591e-01,  4.5329e-01, -5.1144e-01,
           5.5084e-01, -8.6476e-01, -1.2654e-01,  3.0004e-01,  6.8516e-01,
          -3.7960e-01, -2.3610e-03, -1.6699e-02, -2.1314e-01, -2.8445e-01,
          -3.4513e-01, -3.4324e-01, -2.4967e-01,  1.3239e-01, -9.4867e-01,
          -1.5129e+01, -6.0427e-01,  5.6567e-01,  1.6651e-02, -6.0294e-02,
          -3.8575e-01, -8.4208e-01, -5.4401e-01, -6.5214e-01, -8.4426e-01,
           2.6389e-01,  6.4038e-02, -7.1790e-01,  9.3770e-01,  3.5451e-01,
          -5.4737e-01, -1.0393e-03,  1.3554e-01, -6.0948e-01, -1.3363e-01,
          -2.4515e-01, -2.8462e-01, -2.6447e-01,  4.8226e-01,  1.5910e-01,
          -2.0365e+00, -2.0934e-02, -1.5231e-01,  1.8443e-01,  8.7268e-02,
          -9.9519e-01,  5.8023e-01,  1.8951e-02, -9.1640e-01, -1.5140e-01,
          -1.5935e-01,  1.5242e-01, -2.6932e-01, -9.5103e-01,  3.1548e-01,
          -5.1498e-01, -9.0318e-01, -3.9464e-01, -1.9773e-01,  5.7779e-01,
          -2.0494e+00, -2.1752e-02,  7.1961e-01,  6.5953e-01,  3.9062e-01,
           3.3534e-01, -1.8978e-01,  7.3613e-01,  1.8310e-01, -2.8214e-01,
          -1.7986e-01,  3.2193e-01, -1.0608e-01, -5.2542e-01, -1.9458e-01,
          -4.0921e-01,  2.2369e-01,  2.2311e-01, -5.8487e-01, -1.0249e-01,
          -6.1244e-01, -1.4699e-01,  3.9656e-01,  7.0173e-01, -2.6571e-01,
           1.2863e-01, -9.9597e-01,  4.1185e-01, -5.0390e-01,  1.8710e-01,
           6.3707e-01,  1.2322e-02,  1.0033e-01,  7.1355e-01, -6.8427e-02,
           2.0072e-01,  3.9953e-01,  4.6771e-01, -1.6556e-02, -6.8507e-01,
           5.6382e-01,  1.7322e-01, -1.8064e-02, -3.7863e-01,  3.0163e-01,
           3.7296e-01, -1.0239e-01, -2.2087e-02,  6.1339e-01,  1.5526e-01,
          -9.6819e-01,  1.1019e-01, -1.1808e-01,  3.1445e-01, -3.2270e-01,
          -7.6286e-01,  3.5425e-01, -1.9528e-01,  5.1242e-01,  4.6548e-01,
          -6.1083e-01, -1.0123e+00, -3.2977e-01,  1.4745e-01, -1.0458e+00,
          -4.2121e-01, -4.2424e-01,  4.5558e-01,  5.8747e-01, -3.1613e-01,
           4.2754e-01,  6.3993e-01, -2.2840e-01, -4.5763e-01, -2.7863e-01,
          -1.3649e-01, -1.0505e+00,  1.5676e-01, -5.7440e-01, -7.4964e-01,
           1.9569e-01,  4.1389e-01, -9.4425e-02,  5.2835e+00,  1.7685e-01,
           2.9219e-01, -8.5532e-01,  3.5894e-01,  3.1275e-01, -1.4534e-01,
          -4.5474e-01, -1.6707e-01, -4.2704e-01, -2.6621e-01, -6.7241e-01,
           7.3905e-01,  1.7540e-01, -2.9537e-01,  8.9223e-01,  9.9420e-02,
          -1.2276e-01, -9.4053e-01, -5.5920e-01, -5.5844e-02,  5.5859e-01,
          -1.0266e-01, -4.8815e-01,  8.1300e-01, -4.6080e-01, -3.5023e-01,
          -6.5827e-02,  3.6993e-01,  3.3747e-01, -9.5957e-02, -7.9319e-01,
          -1.0044e-02, -7.2068e-01, -1.0642e-02,  8.9776e-02, -4.1134e-01,
           3.5837e-01,  5.8001e-01,  1.3355e-01, -2.6359e-01,  7.8351e-01,
           3.2953e-02, -5.7745e-01, -6.7663e-01,  3.0995e-02, -6.1204e-01,
          -5.9076e-01, -5.0614e-01,  5.7643e-01,  1.0008e+00, -1.3270e-01,
          -5.9896e-01, -7.5989e-02, -5.4924e-01, -8.3733e-01,  3.1266e-01,
           1.3918e-01, -3.7781e-03, -4.8147e-01,  5.2728e-01, -4.1273e-01,
           9.5344e-02,  5.0207e-01, -4.1334e-01,  1.2792e-01,  4.6800e-01,
           2.9954e-02, -2.8660e-01,  2.6861e-01,  1.7268e-01, -6.4009e-01,
          -7.0854e-01, -9.2937e+00, -4.1905e-01,  2.3547e-01, -5.7396e-01,
           1.5430e-01,  7.9481e-01, -1.5178e-01,  6.0122e-01,  2.7130e-01,
          -3.6474e-01, -1.9536e-01, -8.6911e-02, -2.5349e-01,  1.5560e-03,
          -1.0912e-01,  2.4510e-01, -4.9589e-01,  3.2070e-01,  5.9336e-01,
           6.0548e-01,  1.4850e-01, -2.8809e-01,  5.2043e-01,  3.5414e-01,
           3.4432e-01, -2.1292e-01,  2.3703e-01, -9.9204e-01, -5.0474e-02,
           5.8424e-01, -6.9862e-02,  8.1304e-02,  5.4982e-01, -7.7700e-01,
          -3.9368e-01, -8.8066e-01,  1.3203e-01,  7.4052e-02, -7.7313e-01,
           4.9348e-01,  3.9261e-02, -1.5013e-01, -4.1452e-01,  2.6893e-01,
           8.0561e-01, -1.6230e-01,  3.0457e-01,  8.5209e-01, -4.5283e-01,
           4.4758e-01,  2.1153e-01,  1.9122e-01, -2.9400e-01, -1.8149e-03,
          -5.6515e-02,  3.3852e-01, -3.4784e-01,  1.8784e-02,  1.3956e-01,
          -3.5183e-01, -2.5924e-01, -5.0715e-01,  4.6042e-01,  1.0259e+00,
           7.2754e-01,  7.8185e-01,  5.2936e-01,  3.4169e-01,  7.6544e-01,
          -8.1090e-01, -4.9539e-02, -4.1586e-01,  7.4894e-01, -6.3757e-01,
           2.5667e-01,  5.9575e-01,  1.3454e-01,  2.6268e-01,  5.0265e-01,
           6.9063e-03, -8.4378e-01,  3.0719e-01, -5.6659e-01, -5.9044e-02,
          -1.4162e-01, -7.0815e-01,  1.5517e-01, -8.1626e-01,  4.3382e-01,
           1.3387e-01,  7.4050e-02, -7.3634e-01, -8.6683e-01,  3.6545e-01,
          -1.9162e-01, -3.5771e-01,  1.8371e-01,  1.8611e-01, -2.9048e-01,
           7.4161e-01, -6.1946e-01, -4.3144e-02,  4.5234e-01, -4.7187e-01,
           6.1956e-02, -3.8249e-01, -2.7119e-01,  1.1821e-01, -7.4638e-01,
          -9.1147e-02, -6.1284e-02,  3.7037e-02,  4.5445e-01, -2.6089e-01,
          -1.4144e-01,  7.7364e-01, -2.6355e-01,  6.2408e-01, -3.1994e-01,
          -5.4861e-01,  1.6871e-01, -6.5583e-01,  2.9526e-01,  1.6848e-01,
          -3.8409e-01,  3.5326e-01,  3.1079e-01,  8.1789e-01,  2.7098e-01,
          -7.3080e-01,  2.1496e-02, -1.5928e-01,  1.7852e-01,  1.5640e-01,
           1.0452e-01, -1.0070e-01, -1.1427e-01,  3.8779e-01, -3.5948e-01,
          -1.8059e-01, -3.9174e-01,  5.6886e-01,  4.9265e-01,  1.5954e-01,
           8.9203e-02,  1.7006e+00,  5.3097e-02, -1.0341e+00, -2.8136e-01,
           2.4944e-01,  6.8839e-01, -9.6698e-02, -1.1488e+00, -5.4622e-01,
           3.4146e-01, -5.2281e-01,  2.8346e-01,  2.1529e-01,  1.3943e+00,
          -6.7269e-01,  9.9402e-02,  1.0060e+00,  6.4180e-01, -1.0933e-01,
           7.2312e-01,  1.2422e-01,  2.9238e-01, -5.7830e-01, -9.3930e-02,
          -1.4249e-01,  2.7422e-01,  1.3893e-01, -6.5254e-02,  4.7121e-01,
           1.2755e+00, -1.4254e-01, -1.2261e+00, -2.8538e-01,  1.1652e-01,
           5.3754e-01, -2.2650e-01, -8.4797e-01, -4.1161e-01, -5.0532e-01,
           1.3985e-01,  7.4497e-02, -4.4024e-01,  6.7407e-02, -4.1831e-02,
           1.0604e+00, -8.6838e-01,  6.4409e-02,  3.0175e-01,  3.4785e-01,
           9.5812e-02,  1.1777e-01,  2.7412e-01,  1.4770e-01,  5.3034e-01,
           9.0887e-02, -2.0598e-01, -1.0444e+00,  5.5126e-01,  5.5259e-01,
          -3.5752e-01, -1.1402e+00, -7.6057e-02,  2.2249e-01,  3.9004e-01,
          -2.6181e-01, -3.7811e-01, -3.3159e-01, -8.0901e-02, -2.6154e-01,
           3.5109e-01, -3.0051e-01, -4.0502e-01,  3.7459e-01, -1.1865e-01,
          -4.4610e-01,  1.6786e-01, -2.5004e-01, -1.5684e-01, -2.9405e-01,
           1.1027e-01, -9.6861e-02,  1.2864e-01, -2.8683e-02,  8.7043e-01,
          -3.2783e-01,  1.3470e-01,  3.6207e-02, -1.8379e-01,  1.6099e-01,
           1.9188e-01,  5.8841e-01,  1.9539e-01, -5.9928e-01,  4.2484e-01,
           1.3820e-01,  6.5911e-01, -1.0275e+00, -2.2975e-01, -1.2084e-01,
          -4.8366e-01,  1.5273e-01,  4.4026e-01,  2.5293e-03, -6.4945e-02,
          -1.0577e-01, -1.1782e-01,  7.1818e-02,  2.5648e-01, -9.5896e-01,
          -3.1243e-01, -8.6928e-01,  8.1597e-02,  3.4098e-01,  6.9092e-01,
          -4.6632e-02,  1.6311e-01,  2.8601e-01, -1.7308e-01, -1.1959e-02,
           1.1438e+00,  1.7281e-01, -8.8783e-01,  3.4216e-01, -7.1350e-02,
          -3.7813e-01,  3.5796e-01, -3.6328e-01,  2.1710e-01,  2.1130e-01,
          -1.4458e-01,  1.5907e-01, -3.9699e-01,  3.5405e-01,  1.0186e-01,
          -8.3606e-02, -5.2517e-02, -8.4341e-01, -6.6619e-03,  3.9718e-01,
          -6.2875e-02,  7.2003e-02, -2.2825e-02, -4.5825e-01, -6.9813e-01,
          -9.2843e-01, -5.1929e-01,  7.3204e-01,  1.2509e+00, -1.7067e-01,
          -7.0838e-01, -9.1830e-01,  8.8002e-02,  4.2317e-01,  5.5825e-01,
          -3.8786e-01, -8.7177e-01, -2.1303e-01,  4.4943e-01, -3.0673e-01,
           3.7002e-01, -7.7388e-01,  7.1990e-02,  7.8776e-01,  7.4956e-02,
          -5.2348e-01, -1.5019e-01,  8.7144e-03, -7.1603e-01, -7.1644e-01,
          -5.4167e-01,  3.8696e-01,  9.1727e-02,  3.6912e-01, -5.2770e-01,
          -3.2090e-01, -4.2872e-01,  5.6781e-01,  2.1204e-01,  2.9984e-01,
           3.6689e-02,  4.2757e-01, -6.6840e-02,  6.1161e-01, -1.2082e-01,
          -1.0982e+00,  3.6305e-01, -5.0004e-01, -1.0481e-01,  1.3113e+00,
          -4.4254e-01,  6.2662e-01, -3.4315e-01, -4.8654e-01,  4.1413e-02,
          -4.9833e+00,  3.8893e-02, -3.4393e-02, -1.7457e-01, -3.9040e-01,
           2.2913e-01,  1.8626e-01,  2.4573e-02,  4.0977e-01, -1.9457e-01,
           4.3656e-01, -5.1829e-02, -5.4883e-01, -4.0841e-01, -3.8384e-01,
           5.8838e-01,  3.6499e-01, -6.1254e-01,  6.7168e-01,  1.6272e-01,
           2.3727e-01, -3.0993e-02,  4.9746e-01, -1.6534e-01, -1.9408e-01,
           8.6395e-02, -3.0598e-01, -5.7015e-01, -3.7614e-02, -1.7181e-01,
           4.5752e-01,  2.1234e-01,  2.3444e-01, -7.2374e-02,  9.5238e-01,
          -3.7161e-01,  6.8616e-01, -4.4355e-01,  1.9432e-01, -2.3774e-01,
          -5.6820e-01, -5.8610e-01, -1.0096e-01, -6.9678e-02,  3.2554e-01,
           1.2490e-01, -7.0842e-01, -4.8841e-01]]]


class LambdaTermsDataset(Dataset):
    def __init__(self, input_sentences_file, main_dir, transform=None, target_transform=None):
        self.main_dir = main_dir
        self.input_sentences = pd.read_csv(input_sentences_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.input_sentences)

    def __getitem__(self, index):
        sentence = self.input_sentences.iloc[index, 0]
        if sentence[0] == '""': sentence = sentence[1]
        if sentence[-1] == '""': sentence = sentence[:-1]
        path = self.input_sentences.iloc[index, 2]
        path = DATA_PATH + path[len("lambdaBERT/data/"):]
        with open(path, 'r') as f:
            lambda_terms = f.readlines()[0].strip()

        # remove the ")" from the lambda_term:
        lambda_terms = lambda_terms.replace(")", "")

        if self.transform:
            sentence = self.transform(sentence)
        if self.target_transform:
            lambda_terms = self.target_transform(lambda_terms)
        
        return sentence, lambda_terms
    
class ShuffledLambdaTermsDataset(Dataset):
    def __init__(self, input_sentences_file, main_dir, transform=None, target_transform=None):
        self.main_dir = main_dir
        self.input_sentences = pd.read_csv(input_sentences_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.input_sentences)

    def __getitem__(self, index):
        sentence = self.input_sentences.iloc[index, 1]
        if sentence[0] == '""': sentence = sentence[1]
        if sentence[-1] == '""': sentence = sentence[:-1]
        path = self.input_sentences.iloc[index, 2]
        path = DATA_PATH + path[len("lambdaBERT/data/"):]
        with open(path, 'r') as f:
            lambda_terms = f.readlines()[0].strip()

        # remove the ")" from the lambda_term:
        lambda_terms = lambda_terms.replace(")", "")

        sent_embs, target_embs, target_tokens, lambda_index_mask, var_index_mask_no, app_index_mask = torch.load(path.replace("txt", "pt"))#create_out_tensor(sentence, lambda_terms)

        #attach the CLS and SEP tokens to the start and end of target_embs?

        if len(target_embs) == 0:
            if "section_6186/4.txt" in path or "section_4652/5.txt" in path or "section_7065/2.txt" in path or "/section_5020/4.txt" in path:
                target_embs = torch.rand(1, 768)
                target_tokens = [102]
                assert len(lambda_index_mask) == 0 
                assert len(var_index_mask_no) == 0
                assert len(app_index_mask) == 0

                lambda_index_mask.append(0)
                var_index_mask_no.append(0)
                app_index_mask.append(0)
            else:
                raise Exception

        
        #sep token for target embedding:
        target_embs = torch.cat([target_embs, torch.tensor(SEP_TOKEN).squeeze(0)], dim=0)
        target_tokens = target_tokens + [102]
        lambda_index_mask.append(0)
        var_index_mask_no.append(0)
        app_index_mask.append(0)
        
        return sent_embs, target_embs, target_tokens, lambda_index_mask, var_index_mask_no, app_index_mask

def shuffled_collate(batch):
    sent_embedding, lambda_term_embedding, lambda_term_tokens, lambda_mask, var_mask, app_mask = zip(*batch)
    
    sent_embedding, lambda_term_embedding, lambda_term_tokens, lambda_mask, var_mask, app_mask = [sent.squeeze(0) for sent in sent_embedding], [lambda_term.squeeze(0) for lambda_term in lambda_term_embedding], [torch.tensor(sent, dtype=torch.float32) for sent in lambda_term_tokens],[torch.tensor(sent, dtype=torch.bool).squeeze(0) for sent in lambda_mask], [torch.tensor(var, dtype=torch.bool).squeeze(0) for var in var_mask], [torch.tensor(app, dtype=torch.bool).squeeze(0) for app in app_mask]

    sent_embedding_batched = pad_sequence(sent_embedding, batch_first=True, padding_value = 0)
    try:
        lambda_term_embedding_batched = pad_sequence(lambda_term_embedding, batch_first=True, padding_value = 15)
        lambda_term_tokens_batched = pad_sequence(lambda_term_tokens, batch_first=True, padding_value = 0)
    except:
        print([lambda_term.shape for lambda_term in lambda_term_embedding])
        raise Exception
    var_mask_batched = pad_sequence(var_mask, batch_first=True, padding_value = 0)
    lambda_mask_batched = pad_sequence(lambda_mask, batch_first=True, padding_value = 0)
    app_mask_batched = pad_sequence(app_mask, batch_first=True, padding_value = 0)

    lambda_pad_mask = lambda_term_embedding_batched == 15
    lambda_term_embedding_batched = lambda_term_embedding_batched.masked_fill(lambda_pad_mask, 0)

    #extend the masks
    # lambda_mask_batched = lambda_mask_batched.unsqueeze(-1).expand(-1, -1, lambda_term_embedding_batched.size(-1))
    # var_mask_batched = var_mask_batched.unsqueeze(-1).expand(-1, -1, lambda_term_embedding_batched.size(-1))
    # app_mask_batched = app_mask_batched.unsqueeze(-1).expand(-1, -1, lambda_term_embedding_batched.size(-1))
    #contract the mask
    lambda_pad_mask = lambda_pad_mask.sum(-1) >= 1

    return sent_embedding_batched, lambda_term_embedding_batched, lambda_term_tokens_batched, var_mask_batched, lambda_mask_batched, app_mask_batched, lambda_pad_mask


def data_init(batch_size, mode=0, shuffled=False):
    
    #load in the tokenizer
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if not shuffled: dataset = LambdaTermsDataset(DATA_PATH + 'input_sentences.csv', DATA_PATH + 'lambda_terms/')
    else: dataset = ShuffledLambdaTermsDataset(DATA_PATH + 'input_sentences.csv', DATA_PATH + 'lambda_terms/')
    #split the datset 70 20 10 split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    if mode == 2:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=shuffled_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=shuffled_collate)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=shuffled_collate)
        return train_dataloader, val_dataloader, test_dataloader
    elif mode == 0:
        train_size += test_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=9, collate_fn=shuffled_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=9, collate_fn=shuffled_collate)

        return train_dataloader, val_dataloader
    else:
        #just one dataloader
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=9, collate_fn=shuffled_collate)
        return train_dataloader



            
