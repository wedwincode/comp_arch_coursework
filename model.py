import torch


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(out_channels), # todo
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # torch.nn.BatchNorm2d(out_channels), # todo
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = torch.nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, c5)

        self.up4 = torch.nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)

        self.up3 = torch.nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)

        self.up2 = torch.nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)

        self.up1 = torch.nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        self.out_conv = torch.nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)