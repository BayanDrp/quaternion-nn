#!/bin/sh
set -e

cd "$(dirname "$0")/.."
PREFIX="$(pwd)"
DATADIR="$PREFIX/data/mnist"
mkdir -p "$DATADIR"

BASE=https://ossci-datasets.s3.amazonaws.com/mnist
FILES="train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz"

cd "$DATADIR"
for f in $FILES; do
    if [ -f "$f" ]; then
        echo "have $f"
    else
        echo "fetching $f ..."
        if command -v curl >/dev/null 2>&1; then
            curl -sSLO "$BASE/$f"
        else
            wget -q "$BASE/$f"
        fi
    fi
done

echo "MNIST data ready in $DATADIR"