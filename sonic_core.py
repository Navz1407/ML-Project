"""Sonic Sleuth core: preprocessing, feature extraction (LFCC / MFCC / CQT) and the CNN.
Follows Alshehri et al., Computers 2024, 13, 256 (Sections 3.2-3.4)."""
import os
import numpy as np
import scipy.fft
import librosa
import torch
import torch.nn as nn

SR = 16_000                 # paper: downsample to 16 kHz
BLOCK_S = 4                 # paper: 4-second blocks
BLOCK = SR * BLOCK_S
N_FFT, HOP = 512, 256
FEATURES = ("lfcc", "mfcc", "cqt")
FEATURE_SHAPE = {"lfcc": (60, 251), "mfcc": (60, 251), "cqt": (84, 251)}
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


# ---------- preprocessing (paper 3.2) ----------
def load_audio(path):
    """Resample to 16 kHz and average channels to mono."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def trim_silence(y, top_db=30):
    """Remove silent stretches. Returns (trimmed audio, kept [start, end] intervals)."""
    iv = librosa.effects.split(y, top_db=top_db)
    if len(iv) == 0:
        return y, np.array([[0, len(y)]])
    return np.concatenate([y[s:e] for s, e in iv]), iv


def make_blocks(y):
    """Cut into 4 s blocks; short/last blocks are padded by *repeating* the audio.
    A tail shorter than 1 s is dropped (unless it is the only block)."""
    blocks, starts = [], []
    for s in range(0, max(len(y), 1), BLOCK):
        seg = y[s:s + BLOCK]
        if len(seg) < BLOCK:
            if blocks and len(seg) < SR:
                break
            seg = np.resize(seg, BLOCK)
        blocks.append(seg.astype(np.float32))
        starts.append(s)
    return blocks, starts


# ---------- features (paper 3.3) ----------
def _linear_fb(n_filters=128):
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    edges = np.linspace(0, SR / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        l, c, r = edges[i:i + 3]
        fb[i] = np.maximum(0, np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)))
    return fb


_LIN_FB = _linear_fb()


def lfcc(y, n=60):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    return scipy.fft.dct(np.log(_LIN_FB @ S + 1e-10), axis=0, type=2, norm="ortho")[:n]


def mfcc(y, n=60):
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=n, n_fft=N_FFT, hop_length=HOP, n_mels=128)


def cqt(y):
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, n_bins=84, bins_per_octave=12))
    return librosa.amplitude_to_db(C, ref=np.max)


def extract(y, feature):
    x = {"lfcc": lfcc, "mfcc": mfcc, "cqt": cqt}[feature](y)
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


# ---------- model (paper 3.4) ----------
class SonicSleuth(nn.Module):
    """3 x (Conv+ReLU -> MaxPool) -> Flatten -> Dense+ReLU -> Dropout(0.1) -> Dense(1) -> sigmoid.
    forward() returns logits; use torch.sigmoid for P(fake)."""

    def __init__(self, in_shape):
        super().__init__()
        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64))
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 1, *in_shape)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.conv(x))


def load_model(feature):
    path = os.path.join(WEIGHTS_DIR, f"{feature}.pt")
    if not os.path.exists(path):
        return None
    m = SonicSleuth(FEATURE_SHAPE[feature])
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()