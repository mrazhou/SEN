# from modules.transformer import PositionalEncoding
import math
import torch
import torch.nn as nn
from utils import load_weight, get_net_type
import os
from modules.RSICCformer import TransDecoder


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class CrossAttention(nn.Module):
    def __init__(self, d_model=512, n_head=4, dropout=0.5, dim_feedforward=1024):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, image, feature):
        attn_out, _ = self.attn(image, feature, feature)
        out = self.norm1(image + self.dropout1(attn_out))
        linear_out = self.linear2(self.dropout(self.relu(self.linear1(out))))
        out = self.norm2(out + self.dropout2(linear_out))
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction_ratio,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction_ratio,
                      in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, feature):
        avg_out = self.mlp(self.avg_pool(image))
        max_out = self.mlp(self.max_pool(image))
        return self.sigmoid(avg_out + max_out)*feature


class CAGD(nn.Module):
    def __init__(self, d_model=512, n_head=4, dropout=0.5):
        super(CAGD, self).__init__()
        self.cross_attn_bef = CrossAttention(d_model, n_head, dropout, dim_feedforward=d_model*2)
        self.cross_attn_aft = CrossAttention(d_model, n_head, dropout, dim_feedforward=d_model*2)

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _self_attn(self, x):
        attn_out, _ = self.attn(x, x, x)
        out = x + self.dropout(attn_out)
        return self.norm(out)

    def forward(self, before, after, feature):
        # feature: seq_len, n, d_model
        # images: sqe_len, n, d_model
        residual = feature  # TODO whether to add residual
        before = self.cross_attn_bef(before, feature)
        after = self.cross_attn_aft(after, feature)
        diff = after - before
        diff = self._self_attn(diff)
        return before, after, diff + residual


class DiffEncoder(nn.Module):
    def __init__(self, num_layers=3, d_model=512, nhead=8, dropout=0.5):
        super(DiffEncoder, self).__init__()
        
        """Shallow Feature Embedding"""
        # down sample images(6, 256, 256) to (6, 8, 8) 256/8=32 when model_stage = 4
        # down sample images(6, 256, 256) to (6, 16, 16) 256/16=16 when model_stage = 3
        stride = 32 if int(os.environ.get('model_stage', 4)) == 4 else 16
        self.embed_img = nn.Conv2d(3, d_model, 1, stride=stride)
        self.channel_attn_bef = ChannelAttention(d_model)
        self.channel_attn_aft = ChannelAttention(d_model)
        self.pe_img = PositionalEncoding(d_model, max_len=16*16)

        """Cross Attention Guided Differentiation (CAGD) Module"""
        self.pe_feat = PositionalEncoding(d_model, max_len=16*16)
        self.blocks = nn.ModuleList([CAGD(d_model, nhead, dropout)
                                     for _ in range(num_layers)])

    def _sfe(self, images, feature):
        """Shallow Feature Embedding"""
        # images: n, 3, 256, 256
        # feature: n, 512, 8, 8
        n, c, h, w = feature.shape
        before, after = self.embed_img(images[:, :3, :, :]), self.embed_img(images[:, 3:, :, :])
        before, after = self.channel_attn_bef(before, feature), self.channel_attn_aft(after, feature)
        # permute to (seq_len, batch_size, d_model)
        before, after = before.view(n, c, -1).permute(2, 0, 1), after.view(n, c, -1).permute(2, 0, 1)
        before, after = self.pe_img(before), self.pe_img(after)
        return before, after

    def forward(self, images, feature):
        before, after = self._sfe(images, feature)

        n, c, h, w = feature.shape  # n, 512, 8, 8 for model_stage = 4 resnet18
        feature = feature.view(n, c, -1).permute(2, 0,1)  # shape to (64, n, 512)
        feature = self.pe_feat(feature)

        for blk in self.blocks:
            before, after, feature = blk(before, after, feature)
        
        return feature  # seq_len, n, d_model = 64, n, 512


class SEN(nn.Module):
    def __init__(self, args):
        """args should contain keys as follow:
            model_stage: int
            proj_channel: int
            ft_layer: int
            encoder_n_layers: int
            vocab_size: int
            n_head: int
            dropout: float
            decoder_n_layers: int
        """
        super().__init__()
        weight_path = args.weight_path
        net_type, d_model, proj = get_net_type(weight_path,
                                               args.model_stage,
                                               args.proj_channel)
        
        """Image Feature Extractor"""
        resnet = load_weight(net_type, 
                             weight_path,
                             args.model_stage,
                             args.ft_layer,
                             in_channel=6)
        self.extractor = nn.Sequential(resnet, proj)

        """Difference Feature Encoder"""
        self.encoder = DiffEncoder(num_layers=args.encoder_n_layers,
                                   d_model=d_model,
                                   nhead=args.n_heads,
                                   dropout=args.dropout)
        
        """Caption Decoder"""
        self.decoder = TransDecoder(feature_dim=d_model,
                                    vocab_size=args.vocab_size,
                                    n_head=args.n_heads,
                                    n_layers=args.decoder_n_layers,
                                    dropout=args.dropout)

    def forward(self, images, captions, cap_lens):
        feature = self.extractor(images)
        feature = self.encoder(images, feature)
        feature = self.decoder(feature, captions, cap_lens)
        return feature # (8*8, n, 512)

if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(encoder_n_layers=3, n_head=8, dropout=0.5, vocab_size=12000, decoder_n_layers=6)
    net = SEN(args)
    print(net)