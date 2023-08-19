import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channel, num_layers = 2):
        super().__init__()
        self.resnets = nn.ModuleList([Resnet(in_channel) for _ in range(num_layers)])
        self.attentions = nn.ModuleList([Transformer(in_channel) for _ in range(num_layers)])
        self.downsample = nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, image, t, text):
        blocks = list(zip(self.resnets, self.attentions))
        output_states = ()
        for i, (resnet, attention) in enumerate(blocks):
            hidden_states = resnet(image)
            hidden_states = attention(hidden_states, text)

            output_states = output_states + (hidden_states, )

            hidden_states = self.downsample(hidden_states)

            output_states = output_states + (hidden_states, )

        return hidden_states, output_states

class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channel, num_layers = 2):
        super().__init__()
        self.resnets = nn.ModuleList([Resnet(in_channel) for _ in range(num_layers)])
        self.attentions = nn.ModuleList([Transformer(in_channel) for _ in range(num_layers)])
        self.upsample = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)

    def forward(self, image, image_res, t, text):
        for resnet, attention in zip(self.resnets, self.attentions):
            res = image_res[-1]
            image_res = image_res[:-1]
            print(image_res[-1].shape)

            image += res



            image = resnet(image)



            image = attention(image, text)

            return self.upsample(image)


class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.nonLinear1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.nonLinear2 = nn.SiLU()
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


    def forward(self, input):
        x = input
        x = self.norm1(x)
        x = self.nonLinear1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonLinear2(x)
        x = self.dropout(x)
        x = self.conv2(x)

        input = self.shortcut(input)
        return x + input

class Transformer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.keys1 = nn.Linear(in_channel, in_channel)
        self.queries1 = nn.Linear(in_channel, in_channel)
        self.values1 = nn.Linear(in_channel, in_channel)

        self.keys2 = nn.Linear(768, in_channel)
        self.queries2 = nn.Linear(in_channel, in_channel)
        self.values2 = nn.Linear(768, in_channel)

        self.norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        self.feedforward = nn.Sequential(
            nn.Linear(in_channel, in_channel * 8),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(in_channel * 8, in_channel)
        )

        self.out = nn.Conv2d(in_channel, in_channel, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, image, text):
        batch = image.shape[0]
        channel = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]

        image = image.reshape(batch, height * width, channel)

        attn1 = F.scaled_dot_product_attention(self.queries1(image), self.keys1(image), self.values1(image))
        image += attn1

        text = self.norm1(text)

        attn2 = F.scaled_dot_product_attention(self.queries2(image), self.keys2(text), self.values2(text))
        image += attn2

        image2 = self.feedforward(image)

        image2 += image

        return self.out(image2.reshape(batch, channel, height, width))


class DownPath(nn.Module):
    def __init__(self, num_layers = 3):
        super().__init__()
        self.blocks = nn.ModuleList([])
        channel = 320
        self.conv_in = nn.Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        for i in range(num_layers):
            self.blocks.append(CrossAttnDownBlock2D(channel))
            channel *= 2

    def forward(self, x, t, text):
        x = self.conv_in(x)
        down_block_res_samples = (x,)
        for block in self.blocks:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x, z = block(x, t, text)
        return x, z


class MidPath(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.path = CrossAttnDownBlock2D(in_channel)

    def forward(self, x, t, text):
        return self.path(x, t, text)

class UpPath(nn.Module):
    def __init__(self, num_layers = 3):
        super().__init__()
        self.blocks = nn.ModuleList([])
        channel = 2560
        for i in range(num_layers):
            self.blocks.append(CrossAttnUpBlock2D(channel))
            channel //= 2
        self.conv_out = nn.ConvTranspose2d(320, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, x_res, t, text):
        for block in self.blocks:
            print(x.shape)
            x = block(x, x_res, t, text)
        x = self.conv_out(x)
        return x
