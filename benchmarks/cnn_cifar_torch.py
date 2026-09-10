import argparse
import os

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Real (single-scalar) CNN on CIFAR-10, mirroring examples/qcnn_cifar.cpp.
#
# The QCNN treats each RGB pixel as a quaternion (w,x,y = R,G,B, z=0) with
# Hamilton-coupled weights. Here the same widths run as a plain 3-channel real
# conv: R,G,B stay real channels, no coupling. Everything else identical
# (same .bin data, order, epochs, batch, lr, loss).
#
# Modes:
#   default  : one-hot 10-class head, MSELoss, argmax eval (quat side: argmax w)
#   --4d     : single 4D class code head (quaternion labels), MSELoss, eval by
#              cosine(argmax dot product) using the same codes as the quaternion
#              example. Mirrors benchmarks/cnn_torch.py --4d on MNIST.
#
# Data: data/cifar10/{train,test}.bin shared with the C++ loader
# (benchmarks/prep_cifar10.py) for exact parity.
# ============================================================================


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
    # identical to examples/qcnn_cifar.cpp: (cos t, sin t, cos 2t, sin 2t)/sqrt2
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
    parser.add_argument("train_size", nargs="?", type=int, default=8000)
    parser.add_argument("--4d", dest="labels_4d", action="store_true",
                        help="4D quaternion class codes instead of one-hot")

    args = parser.parse_args()

    torch.manual_seed(7)

    root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar10")
    train_img, train_lab = load_bin(os.path.join(root, "train.bin"), args.train_size)
    test_img, test_lab = load_bin(os.path.join(root, "test.bin"))

    model = CNN(8, 16, labels_4d=args.labels_4d)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"torch params (real):      {n_params}")
    # QCNN reference (examples/qcnn_cifar.cpp): same widths, quaternion weights
    # (4 reals); conv1 [8,1,3x3], conv2 [16,8,3x3], fc [1024, (10 or 1)].
    head = 1 if args.labels_4d else 10
    qnn_params = (
        4 * 8 * 9 + 4 * 8
        + 4 * 16 * 8 * 9 + 4 * 16
        + 4 * head * 1024 + 4 * head
    )
    print(f"QCNN params (quaternion): {qnn_params}   (4 reals per weight)")
    if args.labels_4d:
        print("mode: 4D quaternion class codes (cos/sin), dot-product eval")
    else:
        print("mode: one-hot classes, argmax eval")

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
                # normalized prediction, argmax over cos similarity to the codes
                u = output / output.norm(dim=1, keepdim=True).clamp_min(1e-6)
                prediction = (u @ codes.t()).argmax(dim=1)
            else:
                prediction = output.argmax(dim=1)
            correct += (prediction == y).sum().item()
            seen += y.size(0)

    print(f"test accuracy: {correct} / {seen} = {100.0 * correct / seen:.2f}%")


if __name__ == "__main__":
    main()