# quaternion-nn

Neural networks over quaternions, in C++17. Hamilton-product layers, quaternion
weight init, component-wise real-valued autograd — zero dependencies in the core.

Think *tinygrad meets Hamilton*: `q = w + xi + yj + zk` as a first-class citizen,
from raw quaternion math up to quaternion transformers.

## Status

Skeleton phase — directories and empty stubs are in place, implementation lands
phase by phase (see [Roadmap](#roadmap)). Nothing trains yet.

## Building

Requires: CMake ≥ 3.14, a C++17 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Empty stub files are skipped by CMake; after filling one in, re-run the
`cmake -S . -B build` configure step so its target gets registered.

## Conventions

- Headers declare (`include/qnn/`), sources define (`src/qnn/`); templates live in headers.
- Everything in `namespace qnn`; C++17; lib compiles `-Wall -Wextra -Werror`.
- Quaternions are templated on scalar type, `float` by default.
- Gradients flow component-wise (real-valued chain rule through the Hamilton
  product); every op gets a finite-difference gradient check.

## Layout

```
include/qnn/
  core/       quaternion, vectors, matrices, tensors, shapes
  autograd/   variable, gradient, computation graph
  nn/         module, parameter, init + layers/ attention/ models/
  optim/      optimizer base, sgd, adam
  loss/       mse, cross_entropy
  functional/ matmul, softmax, convolution, normalization
  io/         tensor and model serialization
src/qnn/      matching implementations
tests/        ctest unit tests (mirror of modules)
examples/     quaternion demo, xor, qmlp/qcnn mnist, qtransformer demo
benchmarks/   quaternion ops, matmul, attention
python/       bindings (phase 5)
docs/         design docs (as phases land)
```

## Roadmap

- **Phase 0** — core quaternion math + golden tests (`i*j=k`, norm, inverse).
- **Phase 1** — vector/matrix/tensor, Hamilton matmul, split activations, MSE.
- **Phase 2** — autograd + finite-difference grad checks + SGD.
- **Phase 3** — Linear, QMLP, Adam, cross-entropy, XOR converges.
- **Phase 4** — conv2d/norm/embedding, MNIST loader, QCNN, benchmarks.
- **Phase 5** — QKV attention, transformer demo, model IO, Python bindings, publish.

## License

MIT — see [LICENSE](LICENSE).
