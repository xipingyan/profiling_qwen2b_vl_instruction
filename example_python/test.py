import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

def test_softmax():
    src = torch.randn(2, 9)

    f = nn.Softmax(dim=1)
    dst = f(src)

    print("src=", src)
    print("dst=", dst)

    sub1=math.exp(src[0][0]) + math.exp(src[0][1]) + math.exp(src[0][2])
    sub2=math.exp(src[0][3]) + math.exp(src[0][4]) + math.exp(src[0][5])
    sub3=math.exp(src[0][6]) + math.exp(src[0][7]) + math.exp(src[0][8])
    print(f"sub1: {math.exp(src[0][0])/(sub1)}")
    print(f"sub2: {math.exp(src[0][0])/(sub1+sub2)}")
    print(f"sub3: {math.exp(src[0][0])/(sub1+sub2+sub3)}")

def decompessed_sdpa(query:torch.tensor, key:torch.tensor, value:torch.tensor, scale=None):
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    S = torch.matmul(query, key.transpose(-2, -1))
    S = S * scale_factor
    P = F.softmax(S, dim=-1)
    O = torch.matmul(P, value)
    return O

def test_sdpa():
    batch_size = 2
    num_heads = 4
    seq_len_q = 10  # Sequence length for queries
    seq_len_kv = 12 # Sequence length for keys and values
    head_dim = 64

    # Create dummy tensors
    if os.path.exists('./query.pt'):
        query = torch.load('./query.pt')
        key = torch.load('./key.pt')
        value = torch.load('./value.pt')
    else:
        query = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len_kv, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len_kv, head_dim)
        torch.save(query, './query.pt')
        torch.save(key, './key.pt')
        torch.save(value, './value.pt')

    attn_output = F.scaled_dot_product_attention(query, key, value)
    print("Basic Attention Output Shape:", attn_output.shape)
    # Expected shape: (batch_size, num_heads, seq_len_q, head_dim)
    attn_output2 = decompessed_sdpa(query, key, value)

    eq = torch.equal(attn_output, attn_output2)
    print(f"The 2 attn output equal = {eq}")
    thr = 0.0001
    isclose = torch.isclose(attn_output, attn_output2, rtol=thr)
    print(f"The 2 attn output isclose = {isclose.all()}, rtol = {thr}")

    if not isclose.all():
        print("== Compare first low:")
        for i in range(3):
            print(f"  [][][][{i}] {attn_output[0][0][0][i]} vs {attn_output2[0][0][0][i]}")

    # # Example: Prevent the first query token from attending to the last key token
    # attn_mask = torch.ones(batch_size, num_heads, seq_len_q, seq_len_kv, dtype=torch.bool)
    # attn_mask[:, :, 0, -1] = False

    # attn_output_mask = F.scaled_dot_product_attention(
    #     query, key, value, attn_mask=attn_mask
    # )
    # print("Attention Output Shape with Mask:", attn_output_mask.shape)

if __name__ == "__main__":
    # test_softmax()
    test_sdpa()