## SWIFTFormer: Spatial-Wavelet Integrated Frequency Transformer
## for Robust Low-Light Image Enhancement
##
## Debashis Das, Suman Kumar Maji
## Department of Computer Science and Engineering, IIT Patna
##
## Architecture components:
##   GICB      : Gradient-Infused Convolutional Block (with GEB)
##   SCSD-MHA  : Spatial-Channel Shifted Depthwise Multi-Head Attention
##                (cosine-similarity transposed attention)
##   SSF-FFN   : Spatial-Spectral Fusion Feed-Forward Network
##                (DWT dual-path + SSAG gate)
##   H-ENC / H-DEC : 4-stage hierarchical encoder-decoder with
##                    pixel-(un)shuffle resampling + refinement stage

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    _HAS_PYTORCH_WAVELETS = True
except ImportError:  # graceful fallback to built-in Haar transform
    _HAS_PYTORCH_WAVELETS = False


##########################################################################
##  Basic operators
##########################################################################

class RMSNorm(nn.Module):
    """Root-Mean-Square Normalization (RMSN), Eq. (9) pre-normalization.

    N_RMS(x) = x / sqrt(mean(x_c^2) + eps) * gamma   (per-channel gamma)
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / rms * self.gamma


class ShiftConv2d(nn.Module):
    """Parameter-free 3x3 spatial channel shifting (the 'S' operator).

    Channels are split into five groups shifted left / right / up / down /
    identity, injecting spatial mixing at zero parameter cost.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        B, C, H, W = x.shape
        g = C // 5
        out = torch.zeros_like(x)
        out[:, 0 * g:1 * g, :, :-1] = x[:, 0 * g:1 * g, :, 1:]    # shift left
        out[:, 1 * g:2 * g, :, 1:] = x[:, 1 * g:2 * g, :, :-1]    # shift right
        out[:, 2 * g:3 * g, :-1, :] = x[:, 2 * g:3 * g, 1:, :]    # shift up
        out[:, 3 * g:4 * g, 1:, :] = x[:, 3 * g:4 * g, :-1, :]    # shift down
        out[:, 4 * g:, :, :] = x[:, 4 * g:, :, :]                 # identity
        return out


class STP(nn.Module):
    """Shift-Then-Pointwise Conv2D."""

    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.shift = ShiftConv2d(in_dim)
        self.pw = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pw(self.shift(x))


class DSC(nn.Module):
    """Depthwise-Separable Conv2D: 3x3 depthwise -> 1x1 pointwise."""

    def __init__(self, in_dim, out_dim, kernel_size=3, bias=False, pw_groups=1):
        super().__init__()
        self.dw = nn.Conv2d(in_dim, in_dim, kernel_size, padding=kernel_size // 2,
                            groups=in_dim, bias=bias)
        self.pw = nn.Conv2d(in_dim, out_dim, kernel_size=1, groups=pw_groups, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


def _even_pad_conv(x, conv, kernel_size, dilation):
    """'same' padding that also supports even kernels (e.g. 2x2)."""
    eff = (kernel_size - 1) * dilation
    pl, pt = eff // 2, eff // 2
    pr, pb = eff - pl, eff - pt
    return conv(F.pad(x, (pl, pr, pt, pb)))


##########################################################################
##  Haar DWT fallback (used when pytorch_wavelets is unavailable)
##########################################################################

class HaarDWT(nn.Module):
    def forward(self, x):
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]
        d = x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) / 2
        lh = (-a - b + c + d) / 2
        hl = (-a + b - c + d) / 2
        hh = (a - b - c + d) / 2
        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    def forward(self, ll, lh, hl, hh):
        a = (ll - lh - hl + hh) / 2
        b = (ll - lh + hl - hh) / 2
        c = (ll + lh - hl - hh) / 2
        d = (ll + lh + hl + hh) / 2
        B, C, H, W = ll.shape
        out = ll.new_zeros(B, C, H * 2, W * 2)
        out[:, :, 0::2, 0::2] = a
        out[:, :, 1::2, 0::2] = b
        out[:, :, 0::2, 1::2] = c
        out[:, :, 1::2, 1::2] = d
        return out


class WaveletTransform(nn.Module):
    """2D single-level DWT / IDWT wrapper.

    Uses the four-tap Daubechies (db4) basis with periodization mode
    (as in the paper) when pytorch_wavelets is installed; otherwise
    falls back to an exact built-in Haar transform.
    """

    def __init__(self, wave='db4'):
        super().__init__()
        if _HAS_PYTORCH_WAVELETS:
            self.dwt = DWTForward(J=1, wave=wave, mode='periodization')
            self.idwt = DWTInverse(wave=wave, mode='periodization')
            self.haar = False
        else:
            self.dwt_op = HaarDWT()
            self.idwt_op = HaarIDWT()
            self.haar = True

    def decompose(self, x):
        if self.haar:
            return self.dwt_op(x)                       # ll, lh, hl, hh
        ll, highs = self.dwt(x)
        lh, hl, hh = highs[0][:, :, 0], highs[0][:, :, 1], highs[0][:, :, 2]
        return ll, lh, hl, hh

    def reconstruct(self, ll, lh, hl, hh):
        if self.haar:
            return self.idwt_op(ll, lh, hl, hh)
        highs = [torch.stack([lh, hl, hh], dim=2)]
        return self.idwt((ll, highs))


##########################################################################
##  GEB: Gradient Extraction Block  (Eqs. 6-8)
##########################################################################

class GEB(nn.Module):
    """Multi-directional (h / v / d1 / d2) Sobel-style gradient extraction.

    G(i) = sqrt( sum_k [grad_k(i)]^2 ),  k in {h, v, d1, d2}
    """

    def __init__(self, in_channels=3, out_channels=48):
        super().__init__()
        kh = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kv = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        kd1 = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]])
        kd2 = torch.tensor([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]])
        kernels = torch.stack([kh, kv, kd1, kd2]).unsqueeze(1)   # 4 x 1 x 3 x 3
        self.register_buffer('kernels', kernels.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        grads = F.conv2d(x, self.kernels, padding=1, groups=self.in_channels)
        grads = grads.view(B, C, 4, H, W)
        mag = torch.sqrt(torch.sum(grads ** 2, dim=2) + 1e-6)    # Eq. (6)
        return self.proj(mag)


class GICB(nn.Module):
    """Gradient-Infused Convolutional Block  (Eq. 5).

    F0 = fuse( LeakyReLU(Conv3x3(i))  (.)  G(i) ),   (.) = concatenation
    """

    def __init__(self, in_channels=3, dim=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.geb = GEB(in_channels, dim)
        self.fuse = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.conv(x), self.geb(x)], dim=1))


##########################################################################
##  SCSD-MHA  (Eqs. 9-12)
##########################################################################

class SCSD_MHA(nn.Module):
    """Spatial-Channel Shifted Depthwise Multi-Head Attention.

    Q/K/V projections via the cascaded {S, P, D, P} operator chain
    (shift -> pointwise -> depthwise -> pointwise), attention computed
    over the transposed (channel) dimension with L2-normalized
    cosine similarity for illumination-invariant affinity estimation.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # {S, P, D, P} cascade producing Q, K, V jointly
        self.stp = STP(dim, dim * 3, bias=bias)                       # S -> P
        # per-stream (Q/K/V) grouped pointwise keeps projection cost O(d)
        self.dsc = DSC(dim * 3, dim * 3, kernel_size=3, bias=bias, pw_groups=3)  # D -> P
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dsc(self.stp(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2-normalized cosine similarity, Eq. (10)-(11)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature   # transposed map A'
        attn = attn.softmax(dim=-1)                            # Eq. (12)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


##########################################################################
##  SSF-FFN  (Eqs. 13-21)
##########################################################################

class SpectralBranch(nn.Module):
    """Wavelet-domain dual-manifold processing.

    ISS pathway (Y_LL): hierarchical dilated conv cascade pi = {3, 2, 1}
        3x3 (d=3) -> 2x2 (d=2) -> 1x1 (d=1) -> rho 1x1 -> PReLU   (Eqs. 15-16)
    SDS pathway (Y_HF = [Y_LH || Y_HL || Y_HH]): depthwise separable
        hierarchy omega 5x5 -> 3x3 -> 2x2 -> PReLU                (Eq. 17)
    Reconstruction: X_spectro = IDWT(F_LF, F_HF)                  (Eq. 18)
    """

    def __init__(self, dim, wave='db4'):
        super().__init__()
        self.wt = WaveletTransform(wave)

        # Low-frequency (ISS) pathway -- dilation cascade pi = {3, 2, 1}
        # (dilated spatial filtering kept depthwise for parameter parsimony;
        #  cross-channel mixing is delegated to the 1x1 stages)
        self.lf_d3 = nn.Conv2d(dim, dim, 3, dilation=3, groups=dim, bias=False)
        self.lf_d2 = nn.Conv2d(dim, dim, 2, dilation=2, groups=dim, bias=False)
        self.lf_d1 = nn.Conv2d(dim, dim, 1, dilation=1, bias=False)
        self.lf_rho = nn.Conv2d(dim, dim, 1, bias=False)     # rho 1x1 recalibration
        self.lf_act = nn.PReLU(dim)

        # High-frequency (SDS) pathway -- depthwise hierarchy omega
        self.hf_w5 = nn.Conv2d(dim * 3, dim * 3, 5, padding=2, groups=dim * 3, bias=False)
        self.hf_w3 = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.hf_w2 = nn.Conv2d(dim * 3, dim * 3, 2, groups=dim * 3, bias=False)
        self.hf_pw = nn.Conv2d(dim * 3, dim * 3, 1, groups=3, bias=False)
        self.hf_act = nn.PReLU(dim * 3)

    def forward(self, x):
        _, _, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        ll, lh, hl, hh = self.wt.decompose(x)                # Eq. (13)

        # ISS: low-frequency illumination-semantic stream
        f = _even_pad_conv(ll, self.lf_d3, 3, 3)
        f = _even_pad_conv(f, self.lf_d2, 2, 2)
        f = self.lf_d1(f)
        f_lf = self.lf_act(self.lf_rho(f))                   # Eqs. (15)-(16)

        # SDS: high-frequency structural-detail stream
        y_hf = torch.cat([lh, hl, hh], dim=1)                # Eq. (14)
        g = self.hf_w5(y_hf)
        g = self.hf_w3(g)
        g = _even_pad_conv(g, self.hf_w2, 2, 1)
        f_hf = self.hf_act(self.hf_pw(g))                    # Eq. (17)

        lh_o, hl_o, hh_o = f_hf.chunk(3, dim=1)
        out = self.wt.reconstruct(f_lf, lh_o, hl_o, hh_o)    # Eq. (18)
        return out[:, :, :H, :W]


class SSF_FFN(nn.Module):
    """Spatial-Spectral Fusion Feed-Forward Network with SSAG gating.

    Spatial branch  : shift -> pointwise -> depthwise -> pointwise -> GELU (Eq. 19)
    Spectral branch : DWT dual-manifold processing (SpectralBranch)
    SSAG            : G = sigma(W_p(|X_spa . X_spec|));
                      out = G * X_spa + (1 - G) * X_spec              (Eqs. 20-21)
    """

    def __init__(self, dim, ffn_expansion_factor=2.0, wave='db4', bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)

        # spatial stream: S -> P -> D -> P + GELU
        self.spatial = nn.Sequential(
            ShiftConv2d(dim),
            nn.Conv2d(dim, hidden, 1, bias=bias),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=bias),
            nn.Conv2d(hidden, dim, 1, bias=bias),
            nn.GELU(),
        )

        # spectral stream
        self.spectral = SpectralBranch(dim, wave=wave)

        # Spectro-Spatial Attention Gate
        self.ssag_proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        x_spa = self.spatial(x)
        x_spec = self.spectral(x)

        f_d = torch.abs(x_spa * x_spec)                       # interaction strength
        g = torch.sigmoid(self.ssag_proj(f_d))                # G in [0, 1]^{HxWxC}
        fused = g * x_spa + (1.0 - g) * x_spec                # SSAG, Eq. (21)
        return self.project_out(fused)                        # Eq. (20) (+X outside)


##########################################################################
##  Transformer Block
##########################################################################

class TransformerBlock(nn.Module):
    """X -> RMSN -> SCSD-MHA -> (+) -> RMSN -> SSF-FFN -> (+)"""

    def __init__(self, dim, num_heads, ffn_expansion_factor=2.0, wave='db4', bias=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SCSD_MHA(dim, num_heads, bias=bias)
        self.norm2 = RMSNorm(dim)
        self.ffn = SSF_FFN(dim, ffn_expansion_factor, wave=wave, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
##  Resampling (pixel-unshuffle / pixel-shuffle)
##########################################################################

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
##  SWIFTFormer
##########################################################################

class SWIFTFormer(nn.Module):
    """Spatial-Wavelet Integrated Frequency Transformer.

    Args:
        in_channels:  input image channels (3)
        out_channels: output image channels (3)
        dim:          base embedding dimension C
        num_blocks:   Transformer blocks per encoder/decoder stage
        num_refinement_blocks: blocks in the refinement stage T_r''
        heads:        attention heads per stage
        ffn_expansion_factor: hidden expansion of SSF-FFN spatial stream
        wave:         wavelet basis ('db4' as in the paper; 'haar' fallback)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=36,
                 num_blocks=(2, 3, 3, 4),
                 num_refinement_blocks=4,
                 heads=(1, 2, 4, 8),
                 ffn_expansion_factor=2.85,
                 wave='db4',
                 bias=False):
        super().__init__()

        self.gicb = GICB(in_channels, dim)

        def stage(d, n, h):
            return nn.Sequential(*[
                TransformerBlock(d, h, ffn_expansion_factor, wave, bias)
                for _ in range(n)])

        # ---------------- H-ENC ----------------
        self.encoder1 = stage(dim, num_blocks[0], heads[0])
        self.down1 = Downsample(dim)                                    # C   -> 2C
        self.encoder2 = stage(dim * 2, num_blocks[1], heads[1])
        self.down2 = Downsample(dim * 2)                                # 2C  -> 4C
        self.encoder3 = stage(dim * 4, num_blocks[2], heads[2])
        self.down3 = Downsample(dim * 4)                                # 4C  -> 8C
        self.encoder4 = stage(dim * 8, num_blocks[3], heads[3])

        # ---------------- H-DEC ----------------
        self.up4 = Upsample(dim * 8)                                    # 8C -> 4C
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder3 = stage(dim * 4, num_blocks[2], heads[2])

        self.up3 = Upsample(dim * 4)                                    # 4C -> 2C
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder2 = stage(dim * 2, num_blocks[1], heads[1])

        self.up2 = Upsample(dim * 2)                                    # 2C -> C
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.decoder1 = stage(dim, num_blocks[0], heads[0])

        # ---------------- Refinement T_r'' ----------------
        self.reduce_r = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.refinement = stage(dim, num_refinement_blocks, heads[0])

        # residual reconstruction, Eq. (4)
        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    def check_image_size(self, x, factor=8):
        _, _, h, w = x.shape
        ph = (factor - h % factor) % factor
        pw = (factor - w % factor) % factor
        return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w

    def forward(self, inp):
        inp_pad, H, W = self.check_image_size(inp)

        f0 = self.gicb(inp_pad)                       # Eq. (5)

        # H-ENC, Eq. (1)
        f1 = self.encoder1(f0)
        f2 = self.encoder2(self.down1(f1))
        f3 = self.encoder3(self.down2(f2))
        f4 = self.encoder4(self.down3(f3))

        # H-DEC, Eq. (2)
        d3 = self.decoder3(self.reduce3(torch.cat([self.up4(f4), f3], dim=1)))
        d2 = self.decoder2(self.reduce2(torch.cat([self.up3(d3), f2], dim=1)))
        d1 = self.decoder1(self.reduce1(torch.cat([self.up2(d2), f1], dim=1)))

        # refinement, Eq. (3)
        fr = self.refinement(self.reduce_r(torch.cat([d1, f0], dim=1)))

        # residual reconstruction, Eq. (4)
        out = self.output(fr) + inp_pad
        return out[:, :, :H, :W]


if __name__ == '__main__':
    model = SWIFTFormer()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'SWIFTFormer parameters: {n_params / 1e6:.2f} M')
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    print('Input :', tuple(x.shape))
    print('Output:', tuple(y.shape))
