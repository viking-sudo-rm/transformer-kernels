import fasttext
import math
import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

seq_length = 10
d = 300

# # if classic transformer system ...
# def pos_enc(t):
#     w = lambda k: 1/(10000**(2*k//d))
#     pi_2 = np.pi/2
#     return np.matrix([
#         np.sin((i%2) * pi_2 + w(i//2) * t)
#         for i in range(0,d)])
# pos_enc_vecs = [pos_enc(t) for t in range(0, seq_length)]

# if linear
pos_enc_vecs = [[t] for t in range(0, seq_length)]

euclidian_pos_dist = np.zeros((seq_length, seq_length))
for t1 in range(0, seq_length):
    for t2 in range(t1, seq_length):
        # euclidian_pos_dist[t1, t2] = cosine_similarity(pos_enc_vecs[t1], pos_enc_vecs[t2])
        euclidian_pos_dist[t1, t2] = (pos_enc_vecs[t1][0] - pos_enc_vecs[t2][0]) ** 2 / 100.
        euclidian_pos_dist[t2, t1] = euclidian_pos_dist[t1, t2]

pos_embedding_mat = euclidian_pos_dist

model = fasttext.load_model('cc.en.300.bin')

def embedding(block):
    X = torch.zeros((len(block), seq_length, d))
    for i, sentence in enumerate(block):
        wvecs = torch.stack([torch.tensor(model.get_word_vector(word)) for word in sentence[:seq_length]])
        X[i,:wvecs.shape[0],:] = wvecs
    return X


def get_kernel_block(block1, block2, n_layers=2):
    # Blocks have shape [b, n] (values are token indices)
    embed_block1 = embedding(block1)  # [b, n, k]
    embed_block2 = embedding(block2)
    kernel = torch.einsum("ink, jmk -> ijnm", embed_block1, embed_block2)
    A = torch.tensor(pos_embedding_mat).to(torch.float32)

    for _ in range(n_layers):
        # Multiply by the positional embedding matrix.
        kernel = torch.einsum("bn, ijnm -> ijbm", A, kernel)
        kernel = torch.einsum("ijnm, bm -> ijnb", kernel, A)
        # Squared activation.
        kernel = kernel * kernel

    kernel = kernel.mean(dim=(2,3))
    return kernel

if __name__ == '__main__':
    dataset = load_dataset('ptb_text_only')

    sentence_block = dataset['train'][:20]['sentence']
    K = get_kernel_block(sentence_block, sentence_block)
    print(K)
