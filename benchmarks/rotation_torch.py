import argparse
import os

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Real (single-scalar) rotation-estimation MLP, mirroring qcnn_rotation.cpp.
#
# Input: 3 vectors (image of the basis under a random rotation) = 9 floats.
# Target: unit quaternion q (4 floats) -- the rotation itself.
# Real net just outputs 4 floats and must learn the quaternion constraint;
# the QCNN outputs a quaternion natively with Hamilton-coupled weights.
#
# Loss (identical to the C++ side): squared-cosine rotation loss
#     L = 1 - <qhat,q>^2 / (|qhat|^2 |q|^2),  batch mean.
# Exact data parity: same generator (seed 7) writes data/rotation/{train,test}.bin
# if absent, so Colab reproduces byte-identical data to the C++ side.
# ============================================================================


def sample(n, seed=7):
    rng = np.random.default_rng(seed)
    u1 = rng.random(n).astype(np.float32)
    u2 = rng.random(n).astype(np.float32)
    u3 = rng.random(n).astype(np.float32)
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    q = np.stack([w, x, y, z], axis=1)
    w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
    row0 = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    ], axis=-1)
    row1 = np.stack([
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    ], axis=-1)
    row2 = np.stack([
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    R = np.stack([row0, row1, row2], axis=1)
    inputs = np.concatenate([R[:, :, 0], R[:, :, 1], R[:, :, 2]], axis=1)
    return inputs.astype(np.float32), q.astype(np.float32)


def ensure_bins(root):
    a = os.path.join(root, "train.bin")
    b = os.path.join(root, "test.bin")
    if os.path.exists(a) and os.path.exists(b):
        return

    def store(path, inputs, quats):
        with open(path, "wb") as f:
            f.write(np.asarray(inputs.shape[0], dtype=np.int32).tobytes())
            f.write(inputs.tobytes())
            f.write(quats.tobytes())
        print(f"wrote {path}")

    store(a, *sample(12000))
    store(b, *sample(2000))


def load_bin(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        inputs = np.frombuffer(f.read(n * 36), dtype=np.float32).reshape(n, 9)
        targets = np.frombuffer(f.read(n * 16), dtype=np.float32).reshape(n, 4)
    return inputs, targets


class RotationMLP(nn.Module):
    def __init__(self, h1=32, h2=32):
        super().__init__()
        self.l1 = nn.Linear(9, h1)
        self.act = nn.Tanh()
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 4)

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return self.l3(x)


def rotation_loss(out, target):
    num = (out * target).sum(dim=1) ** 2
    den = (out * out).sum(dim=1) * (target * target).sum(dim=1)
    return (1.0 - num / den.clamp_min(1e-12)).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", nargs="?", type=int, default=30)
    parser.add_argument("batch", nargs="?", type=int, default=256)
    parser.add_argument("lr", nargs="?", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(7)

    root = os.path.join(os.path.dirname(__file__), "..", "data", "rotation")
    os.makedirs(root, exist_ok=True)
    ensure_bins(root)

    X, T = load_bin(os.path.join(root, "train.bin"))
    Xt, Tt = load_bin(os.path.join(root, "test.bin"))

    model = RotationMLP(32, 32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"torch params (real):      {n_params}")
    # QCNN reference (examples/qcnn_rotation.cpp): 3->32->32->1 quaternion
    qnn_params = (4 * 3 * 32 + 4 * 32) + (4 * 32 * 32 + 4 * 32) + (4 * 32 + 4)
    print(f"QCNN params (quaternion): {qnn_params}   (4 reals per weight)")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = torch.tensor(X); T = torch.tensor(T)
    Xt = torch.tensor(Xt); Tt = torch.tensor(Tt)

    n = X.shape[0]
    n_batches = n // args.batch
    print(f"training on {n} samples ({n_batches} batches/epoch), lr {args.lr}")
    print(f"device: {device}")

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for b in range(n_batches):
            off = b * args.batch
            x = X[off:off + args.batch].to(device)
            t = T[off:off + args.batch].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = rotation_loss(out, t)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"epoch {epoch + 1}  avg loss {loss_sum / n_batches:.5f}")

        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            model.eval()
            angles = []
            with torch.no_grad():
                for b in range((Xt.shape[0] // args.batch)):
                    off = b * args.batch
                    x = Xt[off:off + args.batch].to(device)
                    t = Tt[off:off + args.batch].to(device)
                    out = model(x)
                    u = out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    dot = torch.abs((u * t).sum(dim=1)).clamp_max(1.0)
                    deg = 2.0 * torch.acos(dot) * 180.0 / np.pi
                    angles.append(deg.cpu().numpy())
            a = np.concatenate(angles)
            print(f"  eval ep {epoch + 1}: mean {a.mean():.3f} deg  "
                  f"median {np.median(a):.3f} deg  "
                  f"<1deg {100 * (a < 1).mean():.2f}%  "
                  f"<5deg {100 * (a < 5).mean():.2f}%  "
                  f"<15deg {100 * (a < 15).mean():.2f}%")


if __name__ == "__main__":
    main()