import json, os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, librosa.display
import soundfile as sf
import streamlit as st
import torch
import sonic_core as sc

st.set_page_config(page_title="Sonic Sleuth", page_icon="🕵️", layout="wide")
st.title("🕵️ Sonic Sleuth — audio deepfake detector")
st.caption("Local re-implementation of the pipeline from Alshehri et al., *Computers* 2024, 13, 256. "
           "Audio never leaves your machine.")


@st.cache_resource
def get_models():
    return {f: sc.load_model(f) for f in sc.FEATURES}


models = {k: v for k, v in get_models().items() if v is not None}

with st.sidebar:
    st.header("Settings")
    top_db = st.slider("Silence threshold (dB below peak)", 20, 50, 30)
    thr = st.slider("Decision threshold on P(fake)", 0.05, 0.95, 0.50, 0.05)
    max_plots = st.slider("Max blocks to draw in detail", 1, 20, 6)
    st.header("Loaded models")
    for f in sc.FEATURES:
        mp = os.path.join(sc.WEIGHTS_DIR, f"{f}_metrics.json")
        if f in models:
            extra = ""
            if os.path.exists(mp):
                m = json.load(open(mp)); extra = f" — test acc {m['accuracy']:.1%}, EER {m['eer']:.3f}"
            st.success(f"{f.upper()}{extra}")
        else:
            st.warning(f"{f.upper()}: no weights")

up = st.file_uploader("Upload an audio file", type=["wav", "flac", "mp3", "ogg", "m4a"])
if up is None:
    st.info("Upload a speech recording to start. The app will walk through every processing step.")
    st.stop()

with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as tmp:
    tmp.write(up.getvalue()); path = tmp.name
st.audio(up.getvalue())
COLORS = plt.cm.tab10.colors


def wave_fig(y, h=1.8):
    fig, ax = plt.subplots(figsize=(11, h))
    ax.plot(np.arange(len(y)) / sc.SR, y, lw=0.5, color="#555")
    ax.set_xlabel("seconds"); ax.set_ylabel("amp"); ax.margins(x=0)
    return fig, ax


# ---------------- Step 1 ----------------
with st.status("Step 1 · Load, resample to 16 kHz, average to mono", expanded=True) as s:
    y = sc.load_audio(path)
    try:
        info = sf.info(path); orig = f"{info.samplerate} Hz, {info.channels} ch"
    except Exception:
        orig = "n/a"
    c = st.columns(3)
    c[0].metric("Original", orig); c[1].metric("Now", f"{sc.SR} Hz, mono"); c[2].metric("Duration", f"{len(y)/sc.SR:.2f} s")
    fig, _ = wave_fig(y); st.pyplot(fig); plt.close(fig)
    s.update(state="complete")

if len(y) < sc.SR // 2:
    st.error("Audio is shorter than 0.5 s — too short to analyse."); st.stop()

# ---------------- Step 2 ----------------
with st.status("Step 2 · Trim silence", expanded=True) as s:
    yt, iv = sc.trim_silence(y, top_db)
    fig, ax = wave_fig(y)
    prev = 0
    for a, b in iv:
        if a > prev: ax.axvspan(prev / sc.SR, a / sc.SR, color="red", alpha=0.25)
        prev = b
    if prev < len(y): ax.axvspan(prev / sc.SR, len(y) / sc.SR, color="red", alpha=0.25)
    st.pyplot(fig); plt.close(fig)
    st.write(f"Red = removed silence. Kept **{len(yt)/sc.SR:.2f} s** of {len(y)/sc.SR:.2f} s.")
    s.update(state="complete")

# ---------------- Step 3 ----------------
with st.status("Step 3 · Slice into 4-second blocks", expanded=True) as s:
    blocks, starts = sc.make_blocks(yt)
    fig, ax = wave_fig(yt)
    for i, st0 in enumerate(starts):
        ax.axvspan(st0 / sc.SR, min(st0 + sc.BLOCK, len(yt)) / sc.SR, color=COLORS[i % 10], alpha=0.2)
        ax.text(st0 / sc.SR + 0.1, ax.get_ylim()[1] * 0.8, f"#{i+1}", fontsize=9)
    st.pyplot(fig); plt.close(fig)
    pad = " The last block was shorter than 4 s and was padded by repeating the audio." if len(yt) % sc.BLOCK or len(yt) < sc.BLOCK else ""
    st.write(f"**{len(blocks)} block(s)**.{pad}")
    s.update(state="complete")

# ---------------- Step 4 ----------------
with st.status("Step 4 · Block-by-block analysis (features, pitch, rhythm)", expanded=True) as s:
    st.caption("Spectrograms (LFCC / MFCC / CQT) are what the CNN actually sees. Pitch and rhythm are shown "
               "for context only — they are not inputs to the classifier in the paper.")
    feats = {f: [] for f in sc.FEATURES}
    prog = st.progress(0.0)
    for i, b in enumerate(blocks):
        for f in sc.FEATURES:
            feats[f].append(sc.extract(b, f))
        prog.progress((i + 1) / len(blocks))
        if i >= max_plots:
            continue
        f0 = librosa.yin(b, fmin=65, fmax=500, sr=sc.SR, frame_length=1024, hop_length=sc.HOP)
        rms = librosa.feature.rms(y=b, frame_length=1024, hop_length=sc.HOP)[0]
        n = min(len(f0), len(rms)); f0, rms = f0[:n].copy(), rms[:n]
        f0[(rms < 0.1 * rms.max()) | (f0 > 490)] = np.nan             # keep voiced frames only
        env = librosa.onset.onset_strength(y=b, sr=sc.SR, hop_length=sc.HOP)
        peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sc.SR, hop_length=sc.HOP)
        t = librosa.frames_to_time(np.arange(n), sr=sc.SR, hop_length=sc.HOP)
        te = librosa.frames_to_time(np.arange(len(env)), sr=sc.SR, hop_length=sc.HOP)

        fig = plt.figure(figsize=(13, 6.2)); gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.18)
        ax = fig.add_subplot(gs[0, 0]); ax.plot(np.arange(len(b)) / sc.SR, b, lw=0.5, color=COLORS[i % 10]); ax.set_title("Waveform", fontsize=9)
        ax = fig.add_subplot(gs[1, 0]); ax.plot(t, f0, ".", ms=2, color="#c0392b"); ax.set_ylabel("Hz"); ax.set_xlim(0, sc.BLOCK_S)
        ax.set_title(f"Pitch (f0): median {np.nanmedian(f0):.0f} Hz, std {np.nanstd(f0):.0f} Hz, voiced {np.mean(~np.isnan(f0)):.0%}"
                     if np.any(~np.isnan(f0)) else "Pitch (f0): no voiced frames", fontsize=9)
        ax = fig.add_subplot(gs[2, 0]); ax.plot(te, env, color="#2980b9"); ax.vlines(te[peaks], 0, env.max(), color="k", alpha=0.4, lw=0.8)
        ax.set_xlim(0, sc.BLOCK_S); ax.set_xlabel("seconds"); ax.set_title(f"Rhythm: {len(peaks)/sc.BLOCK_S:.1f} onsets/s (lines = onsets)", fontsize=9)
        for r, f in enumerate(sc.FEATURES):
            ax = fig.add_subplot(gs[r, 1]); x = feats[f][i]
            ax.imshow(x, origin="lower", aspect="auto", cmap="magma", extent=[0, sc.BLOCK_S, 0, x.shape[0]])
            ax.set_title(f"{f.upper()} {x.shape[0]}×{x.shape[1]} (CNN input)", fontsize=9)
        with st.expander(f"Block #{i+1} — {starts[i]/sc.SR:.1f}s to {min(starts[i]+sc.BLOCK, len(yt))/sc.SR:.1f}s", expanded=(i == 0)):
            st.pyplot(fig)
        plt.close(fig)
    if len(blocks) > max_plots:
        st.write(f"Drew the first {max_plots} blocks; all {len(blocks)} were analysed.")
    s.update(state="complete")

# ---------------- Step 5 ----------------
with st.status("Step 5 · CNN inference on every block", expanded=True) as s:
    if not models:
        st.warning("No trained weights found in `weights/`. Run `train.py` first (see README). "
                   "Steps 1–4 above work without a model, but no verdict can be given.")
        s.update(state="error"); st.stop()
    P = {}
    for f, m in models.items():
        x = torch.from_numpy(np.stack(feats[f]))[:, None]
        with torch.no_grad():
            P[f] = torch.sigmoid(m(x)).squeeze(1).numpy()
    use = [f for f in ("cqt", "lfcc") if f in P] or list(P)          # paper's best ensemble: CQT + LFCC
    ens = np.mean([P[f] for f in use], axis=0)
    rows = {"Block": [f"#{i+1}" for i in range(len(blocks))]}
    for f in P: rows[f"P(fake) {f.upper()}"] = np.round(P[f], 3)
    rows[f"Ensemble ({'+'.join(u.upper() for u in use)})"] = np.round(ens, 3)
    st.dataframe(rows, hide_index=True)
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.bar(range(len(ens)), ens, color=["#c0392b" if p > thr else "#27ae60" for p in ens])
    ax.axhline(thr, ls="--", color="k", lw=0.8); ax.set_ylim(0, 1); ax.set_ylabel("P(fake)")
    ax.set_xticks(range(len(ens))); ax.set_xticklabels([f"#{i+1}" for i in range(len(ens))])
    st.pyplot(fig); plt.close(fig)
    s.update(state="complete")

# ---------------- Step 6 ----------------
st.subheader("Step 6 · Verdict")
p = float(ens.mean()); votes = int((ens > thr).sum())
if p > thr:
    st.error(f"## 🚩 Likely FAKE  —  P(fake) = {p:.1%}")
else:
    st.success(f"## ✅ Likely REAL  —  P(fake) = {p:.1%}")
st.write(f"{votes} of {len(ens)} blocks voted fake. Final score = mean of the per-block ensemble probabilities.")
st.caption("Reference point: in the paper, accuracy on an unseen dataset was ~82–85 %. Treat this as evidence, not proof.")