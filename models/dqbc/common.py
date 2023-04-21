import torch
import torch.nn as nn
import torch.nn.functional as F

def conv373(c):
    return nn.Sequential(
        conv(c,c,3,1,1),
        conv(c,c,7,1,3,groups=4),
        conv(c,c,3,1,1)
    )

def conv333(c):
    return nn.Sequential(
        conv(c,c,3,1,1),
        conv(c,c,3,1,1),
        conv(c,c,3,1,1)
    )

def conv33(c):
    return nn.Sequential(
        conv(c,c,3,1,1),
        conv(c,c,3,1,1)
    )

class RB(nn.Module):
    def __init__(self,c) -> None:
        super().__init__()
        self.conv2 = nn.Sequential(
            conv(c,c),
            nn.Conv2d(c,c,3,1,1)
        )
        self.act = nn.PReLU(c)
    def forward(self, x):
        return self.act(x+self.conv2(x))
    
def make_conv_block(cfg,c):
    if cfg == 333:
        return conv333(c)
    elif cfg == 33:
        return conv33(c)
    elif cfg == 373:
        return conv373(c)
    elif cfg == 'RBx2':
        return nn.Sequential(RB(c),RB(c))
    else:
        raise NotImplementedError

def mlp(in_c,h_c,out_c):
    return nn.Sequential(
        nn.Conv2d(in_c,h_c,1,1,0),
        nn.PReLU(h_c),
        nn.Conv2d(h_c,out_c,1,1,0)
        )

def convex_upsample(x, mask, upscale=8, kernel=3):
    """ Upsample x [B, C, H/d, W/d] -> [B, C, H, W] using convex combination """
    B, C, H, W = x.shape
    mask = mask.view(B, 1, kernel*kernel, upscale, upscale, H, W)

    mask = torch.softmax(mask, dim=2)
    x = F.unfold(x, [kernel,kernel], padding=kernel//2)
    x = x.view(B, C, kernel*kernel, 1, 1, H, W)
    x = torch.sum(mask * x, dim=2) # B,C,upscale,upscale,H,W
    x = x.permute(0, 1, 4, 2, 5, 3)  # B,C,H,upscale,W,upscale

    return x.reshape(B, C, H*upscale, W*upscale)

def bilinear(x,scale_factor):
    return F.interpolate(x,
                         scale_factor=scale_factor,
                         mode='bilinear',
                         align_corners=False)



def conv(in_ch,out_ch,kernel_size=3,stride=1,padding=1,groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=True, groups=groups),
        nn.PReLU(out_ch))


def deconv(in_planes, out_planes):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )

class Conv2(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2) -> None:
        super().__init__()
        self.conv1 = conv(in_ch, out_ch, 4, stride,1)
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x