# quaternion-nn

Neural networks over quaternions, in C++17. Hamilton-product layers, quaternion
weight init, component-wise autograd — zero dependencies in the core.

Think *tinygrad meets Hamilton*: `q = w + xi + yj + zk` as a first-class citizen,
from raw quaternion math up to quaternion transformers.

## Status

The core engine is complete and trainable. Quaternion math, tensors, Hamilton
matmul, a `linear` layer with verified Hamilton backprop, split activations,
`split_softmax` + `cross_entropy`, and `sgd`/`adam` optimizers all work
together — a 2-layer quaternion MLP learns XOR to 4/4 (loss → 0) end to end.

Implemented so far:

- `core/` — `quaternion<T>` (Hamilton product, conjugate, inverse, norm),
  `shape`, `tensor<T>` (row-major, rank-N), `quaternion_vector`/`matrix` views
- `functional/` — Hamilton `matmul`, `matvec`, `split_softmax` (stable), `softmax`
- `nn/` — xavier/uniform init, split ReLU/sigmoid/tanh, `linear` layer with
  forward + backward + gradient accumulators (OpenMP-threaded)
- `loss/` — component-wise `mse`, `cross_entropy` + `cross_entropy_grad`
- `optim/` — `optimizer` base, `sgd`, `adam` (per-component)
- `data/` — MNIST IDX loader (zlib): 2×2 pixel blocks → quaternions, {N,14,14}

### Results

Quaternion MLP on MNIST (196→64→10, MSE one-hot, SGD lr 0.1, 8 epochs):

- **86.67%** test accuracy — C++17, OpenMP x8, ~3 min (see `examples/mnist.cpp`)
- **86.43%** — PyTorch reference on Tesla T4 (`benchmarks/mnist_torch.py`)

Hand-rolled Hamilton backprop keeps pace with a GPU reference at equal setup.

On the roadmap: conv2d/QCNN, attention/transformer, Python bindings.

## Building

Requires: CMake ≥ 3.14, a C++17 compiler. No external libraries.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build          # 11 tests
./build/example_xor             # trains XOR: accuracy 4/4, loss -> 0
./build/example_mnist           # MNIST: 86.67% test (needs tools/get_mnist.sh)
```

Empty stub files keep their `src/qnn/` slot but are skipped by the build; after
filling one in, re-run the `cmake -S . -B build` configure step so its target
gets registered.

## Conventions

- Headers declare (`include/qnn/`), sources define (`src/qnn/`); templates live in headers.
- Everything in `namespace qnn`; C++17; lib compiles `-Wall -Wextra -Werror`.
- Quaternions are templated on scalar type, `float` by default.
- Gradients flow component-wise (real-valued chain rule through the Hamilton
  product); every op gets a finite-difference gradient check.

## Example

Training loop for a single layer (full version in `examples/xor.cpp`):

```cpp
linear<float> layer(3, 2, true);                 // 3 quats in, 2 quats out
sgd<float> opt(0.05f);
opt.add(layer.weight(), layer.dweight());
opt.add(layer.bias(), layer.dbias());

for (int step = 0; step < 200; ++step) {
    opt.zero_grad();
    tensor<quaternion<float>> y = layer.forward(x);   // B×3 -> B×2
    float loss = mse(y, target);
    // upstream gradient dL/dy, then:
    layer.backward(dy);                               // accumulates dW, db
    opt.step();                                       // w -= lr·grad
}
```

`examples/xor.cpp` chains two of these with a `split_tanh` between layers,
showing the full backward pass through the Hamilton product.

## Layout

```
include/qnn/
  core/       quaternion, vectors, matrices, tensors, shapes
  nn/         module, parameter, init + layers/ models/
  optim/      sgd, adam
  loss/       mse, cross_entropy
  functional/ matmul, softmax, convolution, normalization
  io/         tensor and model serialization
src/qnn/      matching implementations
tests/        ctest unit tests (8, mirror of modules)
examples/     quaternion demo, xor, qmlp/qcnn mnist, qtransformer demo
benchmarks/   quaternion ops, matmul, attention, torch reference cross-checks
python/       bindings (phase 5)
docs/         design docs (as phases land)
```

## Roadmap

- [x] **Phase 0** — core quaternion math + golden tests (`i*j=k`, norm, inverse).
- [x] **Phase 1** — vector/matrix/tensor, Hamilton matmul, split activations, MSE.
- [x] **Phase 2** — component-wise autograd + finite-difference grad checks + SGD.
- [x] **Phase 3** — Linear ✓, Adam ✓, XOR converges 4/4 ✓, split-softmax +
      cross-entropy classification head ✓.
- [~] **Phase 4** — MNIST loader ✓, MNIST example (86.67% test) ✓, OpenMP ✓;
      conv2d/norm/embedding, QCNN, benchmarks next.
- [ ] **Phase 5** — attention/transformer, model IO, Python bindings, publish.

## License

MIT — see [LICENSE](LICENSE).