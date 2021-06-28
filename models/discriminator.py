import torch
import torch.nn as nn


class Discriminator(nn.Module):  # 70x70 PatchGAN
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      padding_mode='reflect'),  # in_channels = 6 т.к. конкатенируем A и B изображения
            nn.LeakyReLU(0.2)
        )  # 70 -> 34

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      padding_mode='reflect'),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2)
        )  # 34 -> 16

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      padding_mode='reflect'),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2)
        )  # 16 -> 7

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1),
                      padding_mode='reflect'),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)
        )  # 7 -> 4

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1),
                      padding_mode='reflect'),
            nn.Sigmoid()
        )  # 4 -> 1 (probability if patch is real or fake)

    def forward(self, x):
        e0_conv = self.conv0(x)
        e1_conv = self.conv1(e0_conv)
        e2_conv = self.conv2(e1_conv)
        e3_conv = self.conv3(e2_conv)
        e4_conv = self.conv4(e3_conv)

        return e4_conv