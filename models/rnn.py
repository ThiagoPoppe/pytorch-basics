import torch
import torch.nn as nn
from torchinfo import summary


class RNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = 'tanh'
    ) -> None:
        super(RNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f'Unsupported activation: {activation}')

    def forward(self, x, h_prev):
        h_new = self.W_ih(x) + self.W_hh(h_prev)
        h_new = self.activation(h_new)
        return h_new


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = 'tanh'
    ) -> None:
        super(RNN, self).__init__()
        
        self.rnn_cell = RNNCell(input_size, hidden_size, activation)

    def forward(self, x, h_0 = None):
        if h_0 is None:
            h_0 = torch.zeros(x.size(1), self.rnn_cell.hidden_size, device=x.device)
        
        h_prev = h_0
        outputs = []
        
        for t in range(x.size(0)):
            h_t = self.rnn_cell(x[t], h_prev)
            outputs.append(h_t)
            h_prev = h_t.clone()
        
        outputs = torch.stack(outputs, dim=0)
        return outputs, h_t


if __name__ == '__main__':
    batch_size = 4
    seq_length = 8
    input_size = 8
    hidden_size = 16

    model = RNN(input_size=input_size, hidden_size=hidden_size, activation='tanh')
    x = torch.randn(seq_length, batch_size, input_size)
    h_0 = torch.zeros(batch_size, hidden_size)
    
    output, h_n = model(x, h_0)
    print('> Input shape:', x.shape)
    print('> Output shape:', output.shape)  # (seq_length, batch_size, hidden_size)
    print('> Final hidden state shape:', h_n.shape)  # (batch_size, hidden_size)
    
    summary(model, input_data=[x, h_0])
