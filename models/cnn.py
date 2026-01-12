import torch
import torch.nn as nn
from typing import Any
from torchinfo import summary


def get_num_features_after_convs(img_size, conv_layers):
    width, height = img_size

    for layer in conv_layers:
        kernel_size = layer.get('kernel_size', 1)
        stride = layer.get('stride', 1)
        padding = layer.get('padding', 0)
        dilation = layer.get('dilation', 1)

        width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    return layer['out_channels'] * width * height


class CNN(nn.Module):
    def __init__(
            self,
            input_channels: int,
            conv_layers: list[dict[str, Any]], # same key as in the Conv2D doc
            activation: nn.Module = nn.ReLU(),
        ) -> None:
        super(CNN, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        for conv_layer in conv_layers:
            layers.append(nn.Conv2d(in_channels, **conv_layer))
            layers.append(activation)
            in_channels = conv_layer['out_channels']
        
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv_layers(x)
        return out
    

if __name__ == '__main__':
    input_channels = 1
    conv_layers = [
        {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    
    img_size = (28, 28)
    x = torch.randn(1, input_channels, img_size[0], img_size[1])

    model = CNN(
        input_channels=input_channels,
        conv_layers=conv_layers,
        activation=nn.ReLU()
    )

    output = model(x)
    print('> Input shape:', x.shape)
    print('> Output shape:', output.shape)

    num_features = get_num_features_after_convs(img_size, conv_layers)
    print(f'> Number of features after conv layers: {num_features}')

    summary(model, input_data=x)
