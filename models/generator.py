import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # ENCODER

        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2)
        )  # 256 -> 128

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2)
        )  # 128 -> 64

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2)
        )  # 64 -> 32

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)
        )  # 32 -> 16

        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)
        )  # 16 -> 8

        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)
        )  # 8 -> 4

        self.enc_conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)
        )  # 4 -> 2

        self.enc_conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)
        )  # 2 -> 1

        # DECODER

        self.dec_conv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.dec_conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.dec_conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )

        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.dec_conv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # ENCODE
        e0_conv = self.enc_conv0(x)
        e1_conv = self.enc_conv1(e0_conv)
        e2_conv = self.enc_conv2(e1_conv)
        e3_conv = self.enc_conv3(e2_conv)
        e4_conv = self.enc_conv4(e3_conv)
        e5_conv = self.enc_conv5(e4_conv)
        e6_conv = self.enc_conv6(e5_conv)
        e7_conv = self.enc_conv7(e6_conv)

        # DECODE
        d7_conv = self.dec_conv7(e7_conv)

        d6_cat = torch.cat((d7_conv, e6_conv), 1)
        d6_conv = self.dec_conv6(d6_cat)

        d5_cat = torch.cat((d6_conv, e5_conv), 1)
        d5_conv = self.dec_conv5(d5_cat)

        d4_cat = torch.cat((d5_conv, e4_conv), 1)
        d4_conv = self.dec_conv4(d4_cat)

        d3_cat = torch.cat((d4_conv, e3_conv), 1)
        d3_conv = self.dec_conv3(d3_cat)

        d2_cat = torch.cat((d3_conv, e2_conv), 1)
        d2_conv = self.dec_conv2(d2_cat)

        d1_cat = torch.cat((d2_conv, e1_conv), 1)
        d1_conv = self.dec_conv1(d1_cat)

        d0_cat = torch.cat((d1_conv, e0_conv), 1)
        d0_conv = self.dec_conv0(d0_cat)

        return d0_conv