import argparse
import os

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Real (single-scalar) CNN on CIFAR-10 -- Colab version.
#
# Mirrors examples/qcnn_cifar.cpp exactly: same widths (conv 8->16, fc), same
# normalized inputs, same data ORDER, one-hot / 4D modes, MSELoss, argmax /
# cosine eval. Run on Colab GPU (or any torch box).
#
# Data parity: reads data/cifar10/{train,test}.bin written by
# benchmarks/prep_cifar10.py. If they are missing, they are created on first
# run from torchvision's CIFAR-10 (identical bytes, so C++ and torch see the
# exact same sample order whether or not local).
#
# Example (matches the C++ 10k runs):
#   python3 cnn_cifar_torch_colab.py 4 128 0.1 10000        # one-hot
#   python3 cnn_cifar_torch_colab.py 4 128 0.1 10000 --4d   # 4D codes
# ============================================================================


def ensure_bins(root):
    """Create train.bin / test.bin from torchvision if not present."""
    a = os.path.join(root, "train.bin")
    b = os.path.join(root, "test.bin")
    if os.path.exists(a) and os.path.exists(b):
        return

    import pickle
    import tarfile

    import torchvision

    torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    tar_path = os.path.join(root, "cifar-10-python.tar.gz")
    tar = tarfile.open(tar_path, "r:gz") if os.path.exists(tar_path) else None
    base = "cifar-10-batches-py"

    def load_batch(path):
        f = tar.extractfile(path) if tar is not None else open(path, "rb")
        d = pickle.load(f, encoding="bytes")
        f.close()
        imgs = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
        labels = np.asarray(d[b"labels"], dtype=np.uint8)
        return imgs, labels

    def member(name):
        if tar is not None:
            return f"{base}/{name}"
        return os.path.join(root, base, name)

    tr = [load_batch(member(f"data_batch_{i}")) for i in range(1, 6)]
    te = load_batch(member("test_batch"))
    if tar is not None:
        tar.close()

    def store(imgs, labels, path):
        with open(path, "wb") as f:
            f.write(np.asarray(imgs.shape[0], dtype=np.int32).tobytes())
            f.write(imgs.tobytes())
            f.write(labels.tobytes())
        print(f"wrote {path}")

    store(np.concatenate([x for x, _ in tr]), np.concatenate([y for _, y in tr]), a)
    store(*te, b)


def normalize(imgs):
    # raw [0,255] uint8 -> per-channel (raw/255 - mean) / std, same constants
    # and ordering as examples/qcnn_cifar.cpp.
    x = imgs.astype(np.float32) / 255.0
    mean = np.array([0.49140, 0.48235, 0.44653], dtype=np.float32)
    std = np.array([0.24705, 0.24352, 0.26159], dtype=np.float32)
    return (x - mean) / std


def load_bin(path, limit=None):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        imgs = np.frombuffer(f.read(n * 3072), dtype=np.uint8).reshape(n, 32, 32, 3)
        labels = np.frombuffer(f.read(n), dtype=np.uint8)
    if limit is not None and limit < n:
        imgs = imgs[:limit]
        labels = labels[:limit]
    return normalize(imgs), labels


def class_codes(n_classes=10):
    k = np.arange(n_classes, dtype=np.float32)
    t = 2.0 * np.pi * k / n_classes
    c = np.stack([np.cos(t), np.sin(t), np.cos(2.0 * t), np.sin(2.0 * t)], axis=1)
    return c / np.sqrt(2.0)


class CNN(nn.Module):
    def __init__(self, out1=8, out2=16, output_size=10, labels_4d=False):
        super().__init__()

        self.labels_4d = labels_4d
        self.conv1 = nn.Conv2d(3, out1, 3, padding=1)   # [B,8,32,32]
        self.act = nn.Tanh()
        self.pool = nn.MaxPool2d(2)                     # [B,8,16,16]
        self.conv2 = nn.Conv2d(out1, out2, 3, padding=1)  # [B,16,16,16]
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out2 * 8 * 8, 4 if labels_4d else output_size)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("epochs", nargs="?", type=int, default=4)
    parser.add_argument("batch", nargs="?", type=int, default=128)
    parser.add_argument("lr", nargs="?", type=float, default=0.1)
    parser.add_argument("train_size", nargs="?", type=int, default=10000)
    parser.add_argument("--out1", type=int, default=8)
    parser.add_argument("--out2", type=int, default=16)
    parser.add_argument("--4d", dest="labels_4d", action="store_true",
                        help="4D quaternion class codes instead of one-hot")

    args = parser.parse_args()

    torch.manual_seed(7)

    root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar10")
    os.makedirs(root, exist_ok=True)
    ensure_bins(root)

    train_img, train_lab = load_bin(os.path.join(root, "train.bin"), args.train_size)
    test_img, test_lab = load_bin(os.path.join(root, "test.bin"))

    model = CNN(args.out1, args.out2, labels_4d=args.labels_4d)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"torch params (real):      {n_params}")
    head = 1 if args.labels_4d else 10
    qnn_params = (
        4 * 8 * 9 + 4 * 8
        + 4 * 16 * 8 * 9 + 4 * 16
        + 4 * head * 1024 + 4 * head
    )
    print(f"QCNN params (quaternion): {qnn_params}   (4 reals per weight)")
    print("mode: " + ("4D quaternion class codes (cos/sin), dot-product eval"
                      if args.labels_4d else "one-hot classes, argmax eval"))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    codes = torch.tensor(class_codes(), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    codes = codes.to(device)

    n = train_img.shape[0]
    n_batches = n // args.batch
    print(f"training on {n} samples ({n_batches} batches/epoch), lr {args.lr}")
    print(f"device: {device}")

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for b in range(n_batches):
            off = b * args.batch
            x = torch.tensor(
                train_img[off:off + args.batch],
                dtype=torch.float32,
            ).permute(0, 3, 1, 2)
            y = torch.tensor(train_lab[off:off + args.batch], dtype=torch.long)

            x = x.to(device)
            y = y.to(device)

            if args.labels_4d:
                target = codes[y]                       # [B,4]
            else:
                target = torch.zeros(x.size(0), 10, device=device)
                target.scatter_(1, y.unsqueeze(1), 1.0)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"epoch {epoch + 1}  avg loss {loss_sum / n_batches:.4f}")

    model.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for b in range((test_img.shape[0] // args.batch)):
            off = b * args.batch
            x = torch.tensor(
                test_img[off:off + args.batch],
                dtype=torch.float32,
            ).permute(0, 3, 1, 2)
            y = torch.tensor(test_lab[off:off + args.batch], dtype=torch.long)
            x, y = x.to(device), y.to(device)
            output = model(x)
            if args.labels_4d:
                u = output / output.norm(dim=1, keepdim=True).clamp_min(1e-6)
                prediction = (u @ codes.t()).argmax(dim=1)
            else:
                prediction = output.argmax(dim=1)
            correct += (prediction == y).sum().item()
            seen += y.size(0)

    print(f"test accuracy: {correct} / {seen} = {100.0 * correct / seen:.2f}%")


if __name__ == "__main__":
    main()