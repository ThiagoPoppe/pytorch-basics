import torch
import torch.nn as nn
from torchinfo import summary
from .attention import MultiHeadAttention
     

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward = None, norm_first = False):
        super().__init__()

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        if self.norm_first:
            x_norm = self.norm1(x)
            attn_outputs, attn_weights = self.mha(x_norm, x_norm, x_norm)
            z = attn_outputs + x

            z_norm = self.norm2(z)
            outputs = self.mlp(z_norm) + z
        else:
            attn_outputs, attn_weights = self.mha(x, x, x)
            z = self.norm1(attn_outputs + x)
            outputs = self.norm2(self.mlp(z) + z)

        return outputs, attn_weights
    

if __name__ == '__main__':
    batch_size = 4
    seq_length = 8
    d_model = 16
    num_heads = 4

    model = TransformerLayer(d_model=d_model, num_heads=num_heads, norm_first=True)
    x = torch.randn(batch_size, seq_length, d_model)

    outputs, attn_weights = model(x)
    print('> Input shape:', x.shape)
    print('> Output shape:', outputs.shape)
    print('> Attention Weights shape:', attn_weights[0].shape)

    summary(model, input_data=x)
    