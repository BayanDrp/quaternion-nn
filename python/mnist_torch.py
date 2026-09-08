import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self, input_size=196, hidden_size=64, output_size=10):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("epochs", nargs="?", type=int, default=8)
    parser.add_argument("hidden", nargs="?", type=int, default=64)
    parser.add_argument("batch", nargs="?", type=int, default=128)
    parser.add_argument("lr", nargs="?", type=float, default=0.1)
    parser.add_argument("train_size", nargs="?", type=int, default=60000)

    args = parser.parse_args()

    torch.manual_seed(7)

    # 28x28 -> 14x14 -> 196 features
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
    ])

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

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
    )

    model = MLP(
        input_size=196,
        hidden_size=args.hidden,
        output_size=10,
    )

    # Same idea as your QNN: MSE + one-hot targets
    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print(f"training on {len(train_set)} samples")
    print(f"device: {device}")

    for epoch in range(args.epochs):

        model.train()
        loss_sum = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # [B, 1, 14, 14] -> [B, 196]
            x = images.view(images.size(0), -1)

            # One-hot target
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

    # Evaluation
    model.eval()

    correct = 0
    seen = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            x = images.view(images.size(0), -1)

            output = model(x)

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