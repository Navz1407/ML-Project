"""Train one Sonic Sleuth CNN per feature type.
Expected layout:  data/real/**/*.wav|flac|mp3   data/fake/**/*.wav|flac|mp3
Usage: python train.py --data data --feature lfcc   (also: mfcc, cqt)"""
import argparse, glob, json, os, random
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import train_test_split
import sonic_core as sc

EXT = ("wav", "flac", "mp3", "ogg", "m4a")


class AudioSet(torch.utils.data.Dataset):
    def __init__(self, items, feature, train):
        self.items, self.feature, self.train = items, feature, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        for k in range(10):                       # skip unreadable files
            path, label = self.items[(i + k) % len(self.items)]
            try:
                y, _ = sc.trim_silence(sc.load_audio(path))
                blocks, _ = sc.make_blocks(y)
                b = random.choice(blocks) if self.train else blocks[0]
                x = sc.extract(b, self.feature)
                return torch.from_numpy(x)[None], torch.tensor(label, dtype=torch.float32)
            except Exception:
                continue
        raise RuntimeError("too many unreadable files")


def evaluate(model, loader, dev):
    model.eval(); P, Y, L = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            lg = model(x.to(dev)).squeeze(1).cpu()
            L += nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="sum").item()
            P.append(torch.sigmoid(lg)); Y.append(y)
    return L / len(loader.dataset), torch.cat(P).numpy(), torch.cat(Y).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--feature", choices=sc.FEATURES, required=True)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--patience", type=int, default=10)
    a = ap.parse_args()

    items = []
    for label, name in ((0, "real"), (1, "fake")):
        for e in EXT:
            items += [(f, label) for f in glob.glob(os.path.join(a.data, name, "**", f"*.{e}"), recursive=True)]
    assert items, f"No audio found under {a.data}/real and {a.data}/fake"
    lab = [l for _, l in items]
    print(f"{len(items)} files | real={lab.count(0)} fake={lab.count(1)}")

    # 8.5 : 1 : 0.5 split (split by file, stratified)
    tr, rest = train_test_split(items, test_size=0.15, stratify=lab, random_state=42)
    va, te = train_test_split(rest, test_size=0.5 / 1.5, stratify=[l for _, l in rest], random_state=42)

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    mk = lambda s, t: torch.utils.data.DataLoader(AudioSet(s, a.feature, t), batch_size=a.batch,
                                                  shuffle=t, num_workers=a.workers)
    ltr, lva, lte = mk(tr, True), mk(va, False), mk(te, False)

    model = sc.SonicSleuth(sc.FEATURE_SHAPE[a.feature]).to(dev)
    opt = torch.optim.Adam(model.parameters())
    n_real = sum(1 for _, l in tr if l == 0); n_fake = len(tr) - n_real
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / max(n_fake, 1), device=dev))  # class weights

    os.makedirs(sc.WEIGHTS_DIR, exist_ok=True)
    out = os.path.join(sc.WEIGHTS_DIR, f"{a.feature}.pt")
    best, bad = float("inf"), 0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for x, y in ltr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x).squeeze(1), y); loss.backward(); opt.step()
            tot += loss.item() * len(y)
        vl, _, _ = evaluate(model, lva, dev)
        print(f"epoch {ep:3d}  train {tot/len(tr):.4f}  val {vl:.4f}")
        if vl < best:
            best, bad = vl, 0; torch.save(model.state_dict(), out)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    model.load_state_dict(torch.load(out, map_location=dev, weights_only=True))
    _, p, y = evaluate(model, lte, dev)
    fpr, tpr, _ = roc_curve(y, p); fnr = 1 - tpr; i = np.argmin(np.abs(fnr - fpr))
    m = dict(feature=a.feature, eer=float((fpr[i] + fnr[i]) / 2),
             accuracy=float(((p > .5) == y).mean()), f1=float(f1_score(y, p > .5)), n_test=len(y))
    print(m); json.dump(m, open(os.path.join(sc.WEIGHTS_DIR, f"{a.feature}_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()