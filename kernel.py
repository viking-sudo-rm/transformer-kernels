import torch

# pos_embedding_mat = 


def get_kernel_block(block1, block2, embedding, pos_embedding_mat, n_layers=2):
    # Blocks have shape [b, n] (values are token indices)
    embed_block1 = embedding(block1)  # [b, n, k]
    embed_block2 = embedding(block2)
    kernel = torch.einsum("ink, jmk -> ijnm", embed_block1, embed_block2)
    A = pos_embedding_mat

    for _ in range(n_layers):
        # Multiply by the positional embedding matrix.
        kernel = torch.einsum("bn, ijnm -> ijbm", A, kernel)
        kernel = torch.einsum("ijnm, bm -> ijnb", kernel, A)
        # Squared activation.
        kernel = kernel * kernel

    kernel = kernel.mean(startdim=2)
    return kernel