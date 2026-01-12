import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class AttentionBlock(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(hidden_size))
        self.Wq = nn.Linear(d_model, hidden_size)
        self.Wk = nn.Linear(d_model, hidden_size)
        self.Wv = nn.Linear(d_model, hidden_size)

    def forward(self, q, k, v):
        dot = torch.bmm(self.Wq(q), self.Wk(k).transpose(1, 2))
        attn_weights = F.softmax(dot / self.scale, dim=1)
        attn_outputs = torch.bmm(attn_weights, self.Wv(v))

        return attn_outputs, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, f'{num_heads} must divide {d_model}'

        hidden_size = d_model // num_heads
        self.Wo = nn.Linear(num_heads * hidden_size, d_model)  # 'num_heads * hidden_size' instead of 'd_model' for didatic purposes

        attention_blocks = []
        for i in range(num_heads):
            attn_block = AttentionBlock(d_model, hidden_size)
            attention_blocks.append(attn_block)

        self.attention_blocks = nn.ModuleList(attention_blocks)

    def forward(self, q, k, v):
        attn_outputs = []
        attn_weights = []

        for attention in self.attention_blocks:
            attn_output, attn_weight = attention(q, k, v)
            attn_outputs.append(attn_output)
            attn_weights.append(attn_weight)

        attn_outputs = torch.cat(attn_outputs, dim=-1)
        attn_outputs = self.Wo(attn_outputs)

        return attn_outputs, attn_weights
    

if __name__ == '__main__':
    batch_size = 4
    seq_length = 8
    d_model = 16
    num_heads = 4

    model = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_length, d_model)
    
    output, attn_weights = model(x, x, x) # using as self-attention
    print('> Input shape:', x.shape)
    print('> Output shape:', output.shape)
    print('> Attention weights shapes:')
    for i, weight in enumerate(attn_weights):
        print(f'  > Head {i}:', weight.shape)

    summary(model, input_data=[x, x, x])
