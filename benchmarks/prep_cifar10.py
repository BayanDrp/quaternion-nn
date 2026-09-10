import os
import pickle
import tarfile

import numpy as np

# Reads CIFAR-10 from a downloaded cifar-10-python.tar.gz and writes flat
# binary dumps shared by both sides of the benchmark (exact data parity):
#   int32 n | n * 3072 bytes (row-major 32x32x3 RGB) | n bytes labels


def load_batch(path):
    f = open(path, "rb") if isinstance(path, str) else path
    d = pickle.load(f, encoding="bytes")
    if isinstance(path, str):
        f.close()
    imgs = d[b"data"]  # [N, 3072] uint8, channel-major (R,G,B planes)
    labels = d[b"labels"]
    imgs = imgs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # [N,32,32,3]
    return imgs.astype(np.uint8), np.asarray(labels, dtype=np.uint8)


def store(imgs, labels, path):
    n = imgs.shape[0]
    with open(path, "wb") as f:
        f.write(np.asarray(n, dtype=np.int32).tobytes())
        f.write(imgs.tobytes())
        f.write(labels.tobytes())
    print(f"{path}: {n} x 32x32x3 ({imgs.nbytes / 1e6:.1f} MB)")


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "data", "cifar10")
    tar_path = os.path.join(root, "cifar-10-python.tar.gz")
    assert os.path.exists(tar_path), f"missing {tar_path}"

    with tarfile.open(tar_path, "r:gz") as tar:
        base = "cifar-10-batches-py"
        train_imgs, train_lab = [], []
        for i in range(1, 6):
            img, lab = load_batch(tar.extractfile(f"{base}/data_batch_{i}"))
            train_imgs.append(img)
            train_lab.append(lab)
        test_img, test_lab = load_batch(tar.extractfile(f"{base}/test_batch"))

    train_imgs = np.concatenate(train_imgs)
    train_lab = np.concatenate(train_lab)

    store(train_imgs, train_lab, os.path.join(root, "train.bin"))
    store(test_img, test_lab, os.path.join(root, "test.bin"))


if __name__ == "__main__":
    main()