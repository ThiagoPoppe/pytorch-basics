import torch
import torch.nn as nn
from torchinfo import summary

class MLP(nn.Module):
    def __init__(
            self,
            layer_sizes: list[int],
            activation: nn.Module = nn.ReLU(),
            bias: bool = True
        ) -> None:
        super(MLP, self).__init__()
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))

            if i < len(layer_sizes) - 2: # no activation after last layer
                layers.append(activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out
    

if __name__ == '__main__':
    model = MLP(layer_sizes=[10, 20, 30, 2])
    x = torch.randn(1, 10)

    output = model(x)
    print('> Input shape:', x.shape)
    print('> Output shape:', output.shape)

    summary(model, input_data=x)
