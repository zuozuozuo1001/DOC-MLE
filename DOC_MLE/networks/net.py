import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_erosion(binary_image, kernel):
    inverted_image = 1 - binary_image
    eroded_image = 1 - F.max_pool2d(inverted_image, kernel_size=kernel, stride=1, padding=kernel // 2)
    return eroded_image


def Weighted_GAP(supp_feat, mask, kernel_size):
    eroded_mask = binary_erosion(mask, kernel_size)
    supp_feat = supp_feat * eroded_mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(eroded_mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class Weighted_GAP_multi_strip(nn.Module):
    def __init__(self, ch):
        super(Weighted_GAP_multi_strip, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=1)
        )
        self.conv1_3_5 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(ch, ch, kernel_size=3, dilation=2, padding=2, stride=1)
        )
        self.conv_heng = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv_shu = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.weight = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.BatchNorm2d(ch),
            nn.Sigmoid()
        )

    def forward(self, supp_feat, mask, kernel_size):
        eroded_mask = binary_erosion(mask, kernel_size)
        supp_feat = supp_feat * eroded_mask
        supp_feat1 = self.conv1(supp_feat)
        supp_feat2 = self.conv1_3(supp_feat)
        supp_feat3 = self.conv1_3_5(supp_feat)
        supp_feat4 = self.conv_heng(supp_feat)
        supp_feat5 = self.conv_shu(supp_feat)
        supp_feat_weight = self.weight(supp_feat1 + supp_feat2 + supp_feat3 + supp_feat4 + supp_feat5) * supp_feat
        feat_h, feat_w = supp_feat_weight.shape[-2:][0], supp_feat_weight.shape[-2:][1]
        area = F.avg_pool2d(eroded_mask, (supp_feat_weight.size()[2], supp_feat_weight.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat_weight, kernel_size=supp_feat_weight.shape[-2:]) * feat_h * feat_w / area
        return supp_feat


class MemoryEfficientMish(nn.Module):
    class MishFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.MishFn.apply(x)


class down_pooling(nn.Module):
    def __init__(self, ch):
        super(down_pooling, self).__init__()
        self.down = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class conv_block0(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block0, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            MemoryEfficientMish(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv4 = nn.Sequential(
            MemoryEfficientMish(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        x4 = self.conv4(x3 + x2 + x1)
        return x4


class conv_block1(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block1, self).__init__()
        self.conv1 = nn.Sequential(
            MemoryEfficientMish(),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            nn.Dropout(p=0.5),
        )
        self.conv3 = nn.Sequential(
            MemoryEfficientMish(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            nn.Dropout(p=0.5),
        )
        self.conv4 = nn.Sequential(
            MemoryEfficientMish(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        x4 = self.conv4(x3 + x2 + x1)
        return x4


class ProjectionHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 32, act: nn.Module = None):
        super().__init__()
        if act is None:
            act = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            act
        )
        self.pred = nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.block[0].weight)
        if self.pred.weight is not None:
            nn.init.kaiming_normal_(self.pred.weight, nonlinearity="sigmoid")
        if self.pred.bias is not None:
            nn.init.zeros_(self.pred.bias)

    def forward(self, x, out_size):
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        h = self.block(x)                 
        y = torch.sigmoid(self.pred(h))   
        return y, h


def orthogonal_loss_from_heads(heads, epoch: int, total_epochs: int, eta: float = 0.1):
    K = len(heads)
    device = next(heads[0].parameters()).device
    if K <= 1:
        return torch.tensor(0.0, device=device)

    lam_time = min(max(epoch, 0) / max(total_epochs, 1), 1.0)
    loss_val = torch.tensor(0.0, device=device)

    for i in range(K):
        Wi = heads[i].block[0].weight.view(heads[i].block[0].weight.shape[0], -1)  # [mid_i, in_i]
        for j in range(i + 1, K):
            Wj = heads[j].block[0].weight.view(heads[j].block[0].weight.shape[0], -1)
            M = torch.matmul(Wi, Wj.t())   # Θ_i Θ_j^T
            frob2 = torch.sum(M * M)       # || · ||_F^2
            lam = lam_time * (1.0 + eta * abs(i - j))
            loss_val = loss_val + lam * frob2

    return loss_val



class UNet_Student_Teacher(nn.Module):
    def __init__(self):
        super(UNet_Student_Teacher, self).__init__()
        self.st_conv1 = conv_block0(1, 32)
        self.st_conv2 = conv_block1(32, 64)
        self.st_conv3 = conv_block1(64, 128)
        self.st_conv4 = conv_block1(128, 128 * 2)
        self.st_conv5 = conv_block1(128 * 2, 128 * 4)

        self.st_pool32 = down_pooling(32)
        self.st_pool64 = down_pooling(64)
        self.st_pool128 = down_pooling(128)
        self.st_pool256 = down_pooling(256)

        self.st_upconv1 = self.upconv(64, 32)
        self.st_upconv2 = self.upconv(128, 64)
        self.st_upconv3 = self.upconv(128 * 2, 128)
        self.st_upconv4 = self.upconv(128 * 4, 128 * 2)

        self.st_conv6 = conv_block1(128 * 4, 128 * 2)
        self.st_conv7 = conv_block1(256, 128)
        self.st_conv8 = conv_block1(128, 64)
        self.st_conv9 = conv_block1(64, 32)

        self.proj_heads = nn.ModuleList([
            ProjectionHead(512, 32), 
            ProjectionHead(256, 32),  
            ProjectionHead(128, 32),  
            ProjectionHead(64,  32),  
            ProjectionHead(32,  32),  
        ])

        self.gap = Weighted_GAP_multi_strip(32)

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)

    def ortho_loss(self, epoch: int, total_epochs: int, eta: float = 0.1):
        return orthogonal_loss_from_heads(self.proj_heads, epoch, total_epochs, eta)

    def forward(self, x, Confin=False, epoch: int = None, total_epochs: int = None, eta: float = 0.1):
        # encoder
        x1 = self.st_conv1(x)
        x2 = self.st_pool32(x1);  x2 = self.st_conv2(x2)
        x3 = self.st_pool64(x2);  x3 = self.st_conv3(x3)
        x4 = self.st_pool128(x3); x4 = self.st_conv4(x4)
        x5 = self.st_pool256(x4); x5 = self.st_conv5(x5)

        # decoder
        x_up4 = self.st_upconv4(x5)
        x_up4 = self.st_conv6(torch.cat([x_up4, x4], dim=1))
        x_up3 = self.st_upconv3(x_up4)
        x_up3 = self.st_conv7(torch.cat([x_up3, x3], dim=1))
        x_up2 = self.st_upconv2(x_up3)
        x_up2 = self.st_conv8(torch.cat([x_up2, x2], dim=1))
        x_up1 = self.st_upconv1(x_up2)
        x_up1 = self.st_conv9(torch.cat([x_up1, x1], dim=1))

        out_size = x_up1.size()[-2:]

        x_pred5, down5 = self.proj_heads[0](x5,    out_size)
        x_pred4, down4 = self.proj_heads[1](x_up4, out_size)
        x_pred3, down3 = self.proj_heads[2](x_up3, out_size)
        x_pred2, down2 = self.proj_heads[3](x_up2, out_size)
        x_pred1, down1 = self.proj_heads[4](x_up1, out_size)

        # prototype
        if Confin:
            mask_1 = torch.where(x_pred1 >= 0.70, 1.0, 0.0)
            prot = self.gap(x_up1, mask_1, kernel_size=7)
        else:
            mask_1 = torch.where(x_pred1 >= 0.55, 1.0, 0.0)
            prot = self.gap(x_up1, mask_1, kernel_size=3)

        if (epoch is not None) and (total_epochs is not None):
            ortho = orthogonal_loss_from_heads(self.proj_heads, epoch, total_epochs, eta)
        else:
            ortho = orthogonal_loss_from_heads(self.proj_heads, 1, 1, eta)

        return x_pred5, x_pred4, x_pred3, x_pred2, x_pred1, down5, down4, down3, down2, down1, prot, ortho



if __name__ == "__main__":
    model = UNet_Student_Teacher()
    x = torch.randn(2, 1, 256, 256)  # B, C=1, H, W
    outputs = model(x, Confin=False, epoch=5, total_epochs=100, eta=0.1)
    (x_pred5, x_pred4, x_pred3, x_pred2, x_pred1,
     down5, down4, down3, down2, down1, prot, ortho) = outputs
    print(x_pred1.shape, down1.shape, prot.shape, ortho.item())
