import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================================
# Fair benchmark vs the quaternion QCNN (examples/qcnn_mnist.cpp).
#
# The ONLY structural difference is quaternion vs real:
#   - QCNN: every weight is a quaternion (4 real scalars, Hamilton coupling).
#   - here: every weight is a single real scalar (4 input channels, no coupling).
# Everything else is identical: architecture widths, input packing, train/test
# order, epochs, batch, SGD, and the loss function.
#
# Param note: quaternion layers carry 4x the real weights (4 scalars per
# weight). That IS the thing being measured -- the rotation structure for 4x
# the capacity.
#
# Loss parity: qnn::loss::mse averages over all N*K*4 quaternion components
# (sum / (size * 4)). A real net averages over N*K*1 components. The fan count
# cancels the divisor (each quaternion output contributes 4 terms at 1/4 scale
# vs 1 term at 1/1), so the per-weight gradient is the SAME -- no lr
# adjustment needed. Same lr as the C++ side.
# ============================================================================


class CNN(nn.Module):
    def __init__(self, out1=14, out2=10, output_size=10):
        super().__init__()

        # Default widths give exactly QCNN_param_count / 4 = 10792 / 4 = 2698
        # real params:   conv1 [4,14,3x3]+14 = 518
        #                conv2 [14,10,3x3]+10 = 1270
        #                fc    [90,10]+10     = 910     (90 = 10*3*3)
        # i.e. torch gets one quarter of the quaternion net's real budget.
        self.conv1 = nn.Conv2d(4, out1, 3, padding=1)   # [B,out1,14,14]
        self.act = nn.Tanh()
        self.pool = nn.MaxPool2d(2)                     # [B,out1,7,7]
        self.conv2 = nn.Conv2d(out1, out2, 3, padding=1)  # [B,out2,7,7]
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out2 * 3 * 3, output_size)  # out2*9 -> 10

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.flatten(x)                             # [B,out2*9]
        return self.fc(x)


def pack_quaternion_components(img):
    # [B,1,28,28] -> [B,4,14,14]
    # Exact same 2x2 packing the C++ loader uses to build quaternions:
    #   w=(2r,2c), x=(2r,2c+1), y=(2r+1,2c), z=(2r+1,2c+1)
    img = img.squeeze(1)                                # [B,28,28]
    w = img[:, 0::2, 0::2]
    x = img[:, 0::2, 1::2]
    y = img[:, 1::2, 0::2]
    z = img[:, 1::2, 1::2]
    return torch.stack((w, x, y, z), dim=1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("epochs", nargs="?", type=int, default=6)
    parser.add_argument("batch", nargs="?", type=int, default=128)
    parser.add_argument("lr", nargs="?", type=float, default=0.1)
    parser.add_argument("train_size", nargs="?", type=int, default=16000)
    parser.add_argument("--out1", type=int, default=None)
    parser.add_argument("--out2", type=int, default=None)
    parser.add_argument(
        "--4d",
        dest="labels_4d",
        action="store_true",
        help="4D quaternion labels: class k is a unit vector in R^4 "
        "(rotation encoding). Both QCNN and torch then regress to a single "
        "4D code and predict by argmax dot-product.",
    )

    args = parser.parse_args()

    torch.manual_seed(7)

    # Raw 28x28, no Resize: channels come from the 2x2 quaternion packing.
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    if args.train_size < len(train_set):
        train_set = torch.utils.data.Subset(
            train_set,
            range(args.train_size)
        )

    # No shuffle: the C++ side feeds sequential batches.
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
    )

    # 4D quaternion label codes: class k -> unit vector in R^4
    #   (cos t, sin t, cos 2t, sin 2t) / sqrt(2)   with t = 2*pi*k/10
    # Mirrored exactly in examples/qcnn_mnist_4d.cpp.
    codes = torch.tensor(
        [
            [
                math.cos(2 * math.pi * k / 10),
                math.sin(2 * math.pi * k / 10),
                math.cos(4 * math.pi * k / 10),
                math.sin(4 * math.pi * k / 10),
            ]
            for k in range(10)
        ],
        dtype=torch.float32,
    ) / math.sqrt(2)

    # Widths: --4d uses param-matched defaults (torch ~= QCNN ~= 5,572 real).
    output_size = 4 if args.labels_4d else 10
    if args.out1 is None:
        out1 = 8 if args.labels_4d else 14
    else:
        out1 = args.out1
    if args.out2 is None:
        out2 = 48 if args.labels_4d else 10
    else:
        out2 = args.out2

    model = CNN(out1, out2, output_size)
    n_params = sum(p.numel() for p in model.parameters())

    # QCNN reference (examples/qcnn_mnist.cpp): quat conv1 8, conv2 16,
    # every weight a quaternion (4 real scalars). The head outputs either 10
    # quaternion logits (one-hot mode) or a single quaternion (--4d mode).
    QCNN_C1, QCNN_C2 = 8, 16
    head_out_quat = 1 if args.labels_4d else output_size
    qnn_params = (
        4 * QCNN_C1 * 3 * 3 + 4 * QCNN_C1
        + 4 * QCNN_C2 * QCNN_C1 * 3 * 3 + 4 * QCNN_C2
        + 4 * QCNN_C2 * 9 * head_out_quat + 4 * head_out_quat
    )

    print(f"torch params (real):      {n_params}")
    print(f"QCNN params (quaternion): {qnn_params}   (4 reals per weight)")
    print(f"labels: {'4D quaternion codes' if args.labels_4d else 'one-hot 10'}")

    # MSE, one-hot targets -- same cost as qnn::loss::mse (sum/(size*4)) and
    # the same per-weight gradient magnitude (fan count cancels the mean), so
    # the same SGD lr as the C++ side.
    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print(f"training on {len(train_set)} samples  (lr {args.lr}, same as QCNN)")
    print(f"device: {device}")

    for epoch in range(args.epochs):

        model.train()
        loss_sum = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            x = pack_quaternion_components(images)

            if args.labels_4d:
                target = codes[labels].to(device)         # [B,4]
            else:
                target = torch.zeros(
                    x.size(0),
                    10,
                    device=device,
                )
                target.scatter_(
                    1,
                    labels.unsqueeze(1),
                    1.0,
                )

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)

        print(
            f"epoch {epoch + 1}  "
            f"avg loss {avg_loss:.4f}"
        )

    # Evaluation.
    #   10 outputs: argmax logit (QCNN uses argmax over .w -- same convention).
    #   4D labels:  argmax over dot(prediction, class_code) -- matches the
    #               QCNN's evaluation in examples/qcnn_mnist_4d.cpp.
    model.eval()

    correct = 0
    seen = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            x = pack_quaternion_components(images)

            output = model(x)

            if args.labels_4d:
                pred = torch.nn.functional.normalize(output, dim=1)
                prediction = (pred @ codes.to(device).T).argmax(dim=1)
            else:
                prediction = output.argmax(dim=1)

            correct += (
                prediction == labels
            ).sum().item()

            seen += labels.size(0)

    accuracy = 100.0 * correct / seen

    print(
        f"test accuracy: "
        f"{correct} / {seen} = {accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()