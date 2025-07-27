import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class RSU(nn.Module):
    def __init__(self, height, in_ch, mid_ch, out_ch):
        super(RSU, self).__init__()
        self.height = height
        self.in_conv = ConvBlock(in_ch, out_ch)
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Encoder path
        for i in range(height - 1):
            self.enc_blocks.append(
                ConvBlock(out_ch if i == 0 else mid_ch, mid_ch)
            )

        # Bottom
        self.bottom = ConvBlock(mid_ch, mid_ch)

        # Decoder path
        self.dec_blocks = nn.ModuleList()
        for _ in range(height - 1):
            self.dec_blocks.append(ConvBlock(mid_ch * 2, mid_ch))

        self.out_conv = ConvBlock(mid_ch + out_ch, out_ch)

    def forward(self, x):
        x_in = self.in_conv(x)
        enc_feats = []
        hx = x_in

        # Encoder
        for i, block in enumerate(self.enc_blocks):
            hx = block(hx)
            enc_feats.append(hx)
            if i < self.height - 2:  # Don't pool after last encoder block
                hx = self.pool(hx)

        # Bottom
        hx = self.bottom(hx)

        # Decoder
        for i, block in enumerate(self.dec_blocks):
            skip = enc_feats[-(i + 1)]
            hx = F.interpolate(hx, size=skip.shape[2:], 
                              mode='bilinear', 
                              align_corners=True)  # Changed to True for consistency
            hx = torch.cat([hx, skip], dim=1)
            hx = block(hx)

        # Output
        hx = F.interpolate(hx, size=x_in.shape[2:], 
                          mode='bilinear', 
                          align_corners=True)
        hx = torch.cat([hx, x_in], dim=1)
        return self.out_conv(hx)

class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        # Encoder Stages
        self.stage1 = RSU(7, in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU(6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU(5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU(4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU(4, 512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU(4, 512, 256, 512)

        # Decoder Stages
        self.stage5d = RSU(4, 1024, 256, 512)
        self.stage4d = RSU(4, 1024, 128, 256)
        self.stage3d = RSU(5, 512, 64, 128)
        self.stage2d = RSU(6, 256, 32, 64)
        self.stage1d = RSU(7, 128, 16, 64)

        # Side Outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(out_ch * 6, out_ch, 1)

    def forward(self, x):
        # Encoder
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # Decoder
        def cat_and_resize(a, b):
            a = F.interpolate(a, size=b.shape[2:], 
                            mode='bilinear', 
                            align_corners=True)
            return torch.cat([a, b], dim=1)

        hx5d = self.stage5d(cat_and_resize(hx6, hx5))
        hx4d = self.stage4d(cat_and_resize(hx5d, hx4))
        hx3d = self.stage3d(cat_and_resize(hx4d, hx3))
        hx2d = self.stage2d(cat_and_resize(hx3d, hx2))
        hx1d = self.stage1d(cat_and_resize(hx2d, hx1))

        # Side Outputs
        d1 = self.side1(hx1d)
        d2 = F.interpolate(self.side2(hx2d), 
                          size=d1.shape[2:], 
                          mode='bilinear', 
                          align_corners=True)
        d3 = F.interpolate(self.side3(hx3d), 
                          size=d1.shape[2:], 
                          mode='bilinear', 
                          align_corners=True)
        d4 = F.interpolate(self.side4(hx4d), 
                          size=d1.shape[2:], 
                          mode='bilinear', 
                          align_corners=True)
        d5 = F.interpolate(self.side5(hx5d), 
                          size=d1.shape[2:], 
                          mode='bilinear', 
                          align_corners=True)
        d6 = F.interpolate(self.side6(hx6), 
                          size=d1.shape[2:], 
                          mode='bilinear', 
                          align_corners=True)

        # Final fused output
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))
        return d0, d1, d2, d3, d4, d5, d6
