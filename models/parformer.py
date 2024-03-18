import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

class CAPatchEmbedding(nn.Module):
    """ Channel Attention Pacth Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, act_layer=nn.GELU, CAPE=False, embed_dim=768):
        super().__init__()
        conv_kernel  = to_2tuple(patch_size)
        pool_kernel  = to_2tuple(patch_size+(patch_size//2))
        pool_padding = to_2tuple(patch_size//2)

        if patch_size > 3:
            conv_padding = to_2tuple(patch_size//2-1)
        else:
            conv_padding = to_2tuple(patch_size//2)        

        self.conv_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_kernel, stride=stride, 
                              padding=conv_padding)
        if CAPE:
            self.pool    = nn.MaxPool2d(kernel_size=pool_kernel, stride=stride, 
                                padding=pool_padding)   
            self.pw_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
            self.act = act_layer ()
            self.cape = True
        else:
            self.cape = False

        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        if self.cape:
            xr = self.act(self.pw_proj(self.pool(x))) # Channel Attention
            x  = self.conv_proj(x) # Convolutional Patch Embedding
            x  = x + xr
        else:
            x  = self.conv_proj(x) # Convolutional Patch Embedding
        x  = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x  = self.norm(x)
        return x

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2, act_layer=nn.GELU, bias=False, kernel_size=7, padding=3, **kwargs):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act = act_layer()

        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.act(x)
        x = self.pwconv2(x)
        return x

class Attention(nn.Module):
    """ Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, num_heads=None, head_dim=32, qk_scale=None, qkv_bias=True,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q.transpose(-2, -1).contiguous() @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (v @ attn).transpose(1, 2).contiguous().reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ParFormerBlock(nn.Module):

    def __init__(self, dim, act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 tokenmixer1=Attention, tokenmixer2=SepConv, 
                 mlp=Mlp, mlp_ratio=4., layer_scale_init_value=1e-6,
                 drop=0., drop_path=0., shift=False, 
                 block_num=0):
        super().__init__()           

        cs1 = dim//2
        cs2 = dim//2
        self.split_index = (cs1, cs2)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.scale1 = nn.Parameter(layer_scale_init_value*torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale_init_value*torch.ones(dim), requires_grad=True)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if shift and block_num%2:
            self.shift=True
        else:
            self.shift=False
        
        if tokenmixer1 == tokenmixer2:
            self.tokenmixer1 = tokenmixer1(dim=dim, drop=drop)
            self.parallel = False
        else:
            self.tokenmixer1 = tokenmixer1(dim=cs1, drop=drop)
            self.tokenmixer2 = tokenmixer2(dim=cs2, drop=drop)
            self.parallel = True

    def forward(self, x):
        _,_,_,C=x.shape
        x=self.norm1(x)
        if self.parallel:
            x1, x2=torch.split(x, self.split_index, dim=3)
            x1=self.tokenmixer1(x1)
            x2=self.tokenmixer2(x2)
            if self.shift:
                xs=torch.cat((x2,x1), dim=3) #channel join
            else:
                xs=torch.cat((x1,x2), dim=3) #channel join
        else:
            xs=self.tokenmixer1(x)

        x = x + self.scale1*self.drop_path1(xs)
        x = x + self.scale2*self.drop_path2(self.mlp(self.norm2(x)))

        return x

class ParFormer(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, embed_dims=[64, 128, 384, 512], depths=[3, 3, 9, 3],
                 mlp_ratios=[4,4,4,4], num_stages=4, drop_path_rate=0., norm_layer=nn.LayerNorm,  
                 drop_rate=0., tokenmixers1=Attention, tokenmixers2=SepConv, head_block=MlpHead, 
                 layer_scale_init_value=1e-6, head_dropout=0.0, head_init_scale=1.0, token_shift=False,
                 patch_attention=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        if not isinstance(tokenmixers1, (list, tuple)):
            tokenmixers1 = [tokenmixers1] * num_stages

        if not isinstance(tokenmixers2, (list, tuple)):
            tokenmixers2 = [tokenmixers2] * num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.downsamplings=nn.ModuleList()
        self.stages=nn.ModuleList()
        for i in range(num_stages):

            patch_embed = CAPatchEmbedding(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=3 if i == 0 else embed_dims[i - 1],
                                            CAPE=patch_attention,
                                            embed_dim=embed_dims[i])
            self.downsamplings.append(patch_embed) 
            
            block = nn.Sequential(
                *[ParFormerBlock(dim=embed_dims[i], tokenmixer1=tokenmixers1[i], tokenmixer2=tokenmixers2[i],
                                mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, 
                                shift=token_shift, layer_scale_init_value=layer_scale_init_value,
                                block_num=j)
                for j in range(depths[i])])
            self.stages.append(block)
            
            cur += depths[i]

        # classification head
        if head_dropout > 0.0:
            self.head = head_block(embed_dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_block(embed_dims[-1], num_classes)
        self.out_norm = norm_layer(embed_dims[-1])

        self.apply(self._init_weights)
        if head_block == MlpHead:
            self.head.fc1.weight.data.mul_(head_init_scale)
            self.head.fc2.weight.data.mul_(head_init_scale)
            self.head.norm.weight.data.mul_(head_init_scale)
            self.head.fc1.bias.data.mul_(head_init_scale)
            self.head.fc2.bias.data.mul_(head_init_scale)
            self.head.norm.bias.data.mul_(head_init_scale)
            
        else:    
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = MlpHead(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            x = self.downsamplings[i](x)
            x = self.stages[i](x)
            if i != self.num_stages - 1:
                x = x.permute(0, 3, 1, 2).contiguous()
        return self.out_norm(x.mean([1,2]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def ParFormer_B1(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ParFormer(
        embed_dims=[48, 96, 192, 384], depths=[3, 3, 9, 3],
        tokenmixers1=[Attention, Attention, Attention, Attention],
        tokenmixers2=[SepConv, SepConv, SepConv, SepConv],
        token_shift=False, patch_attention=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ParFormer_B2(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ParFormer(
        embed_dims=[64, 128, 320, 512], depths=[3, 3, 9, 3],
        tokenmixers1=[Attention, Attention, Attention, Attention],
        tokenmixers2=[SepConv, SepConv, SepConv, SepConv],
        token_shift=False, patch_attention=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def ParFormer_B3(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ParFormer(
        embed_dims=[64, 128, 320, 512], depths=[3, 12, 18, 3],
        tokenmixers1=[Attention, Attention, Attention, Attention],
        tokenmixers2=[SepConv, SepConv, SepConv, SepConv],
        token_shift=False, patch_attention=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model
