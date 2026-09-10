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

#### Full-quaternion targets experiment

The targets and decoder normally use only the real part (`w`). Instead of real
one-hot targets, classes can be encoded as fixed unit quaternion codes and
decoded by full-quaternion dot-product alignment:

| scheme | test acc |
|---|---|
| real-only one-hot (`w`) | 86.67% |
| cross-polytope codes + dot decode | 86.63% |
| 600-cell codes + dot decode | 85.93% |
| torch T4 (real MLP) | 86.43% |

All statistically tied — at this capacity the real part alone saturates the
task, so the extra imaginary degrees of freedom are headroom, not free
accuracy. They should matter when capacity is the bottleneck (tiny nets or
harder, image-scale tasks like the QCNN). Example:
`examples/mnist_with_not_real_target.cpp`.

#### QCNN vs PyTorch benchmark suite

A full image-model comparison: quaternion QCNN vs a real PyTorch CNN on the
same data, preprocessing, order, seeds, loss, and decode — the only difference
is the weight type (each quaternion weight uses 4 reals). Scripts in
`benchmarks/` (`cnn_torch.py`, `cnn_cifar_torch.py`, `prep_cifar10.py`,
`examples/qcnn_cifar.cpp`, `examples/qcnn_mnist.cpp`, ...).

MNIST (16k train, 6 epochs, batch 128, both sides lr 0.1):

| model | real params | test acc |
|---|---|---|
| QCNN 1-hot (conv 8→16 → 144→10) | 10,792 | 67.07% |
| torch 1-hot, equal width | 2,914 | 85.65% |
| torch 1-hot, equal params | 10,738 | 89.74% |
| QCNN 4-D quaternion codes | 5,572 | 74.98% |
| torch 4-D codes | 5,532 | 86.34% |

CIFAR-10 (10k train, batch 128, per-channel-normalized RGB packed as
`(w,x,y)=(R,G,B)`; QCNN `lr 0.025` 1-hot / `lr 0.01` 4-D vs torch `lr 0.1`):

| model | real params | 4 ep | 16 ep |
|---|---|---|---|
| QCNN 1-hot | 45,992 | 12.72% | — |
| torch 1-hot | 11,642 | 41.47% | — |
| torch 1-hot, equal budget | 45,866 | 45.77% | — |
| QCNN 4-D codes | 9,092 | 16.38% | 20.36% |
| torch 4-D codes | 5,492 | 24.10% | 28.69% |

Rotation estimation (regression to a unit quaternion — the quaternion's home
turf; 12k/2k synthetically generated rotations, batch 256, identical
squared-cosine loss `1 − <q̂,q>²/(|q̂|²|q|²)`, seeds shared):

| model | real params | epochs | mean θ (best) | median θ (best) | <15° |
|---|---|---|---|---|---|
| torch MLP 9→32→32→4 | 1,508 | 30 | 9.3° (7.3°) | 5.3° (4.6°) | 94% |
| QCNN MLP 3→32→32→1 quat | 4,868 | 100, lr 0.3 | 50.4° | 41.9° | 6.9% |

Honest summary: at every matched setup the PyTorch reference wins, often by a
large margin. The quaternion networks train far slower (CIFAR needs ~4× more
epochs to approach torch's 4-epoch results, and high learning rates make them
diverge), and they generalize poorly — training loss falls while test accuracy
stays near chance, the signature of coarse memorization. Even rotation
regression, which should be structurally natural for quaternions, does not
rescue it: torch learns the target function in ~5 epochs (76% <15°); the
quaternion MLP stays near chance after 100. Reproduce with
`benchmarks/gen_rotation_data.py` + `examples/qcnn_rotation.cpp` vs
`benchmarks/rotation_torch.py`.

## Building

Requires: CMake ≥ 3.14, a C++17 compiler. No external libraries.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build          # 16 tests
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

Everything is a `Module<T>` owning `Parameter<T>`s (`{ value, grad }`). Compose
layers, expose their Parameters, and bind the whole model to an optimizer with
one call (full version in `examples/module_parameters.cpp`):

```cpp
class MLP : public Module<float> {
    linear<float> l1, l2;
    // forward(x) / backward(dy)  chain the layers;
    // parameters() / gradients() flatten l1, l2's Parameter lists
};

MLP model;
sgd<float> opt(0.6f);
opt.add(model);                     // binds every Parameter in the model

for (int step = 0; step < 200; ++step) {
    opt.zero_grad();                // clears every Parameter.grad
    tensor<quaternion<float>> y = model.forward(x);
    float loss = mse(y, target);
    // upstream gradient dL/dy, then:
    model.backward(dy);             // accumulates into every Parameter.grad
    opt.step();                     // w -= lr·grad for every Parameter
}
```

`examples/xor.cpp` chains two linear layers with a `split_tanh` between them,
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
benchmarks/   QCNN vs PyTorch cross-checks, rotation bench, data prep
python/       bindings (phase 5)
docs/         design docs (as phases land)
```

## Roadmap

- [x] **Phase 0** — core quaternion math + golden tests (`i*j=k`, norm, inverse).
- [x] **Phase 1** — vector/matrix/tensor, Hamilton matmul, split activations, MSE.
- [x] **Phase 2** — component-wise autograd + finite-difference grad checks + SGD.
- [x] **Phase 3** — Linear ✓, Adam ✓, XOR converges 4/4 ✓, split-softmax +
      cross-entropy classification head ✓.
- [x] **Phase 3** — Linear ✓, Adam ✓, XOR converges 4/4 ✓, split-softmax +
      cross-entropy classification head ✓.
- [x] **Phase 4** — MNIST loader ✓, MNIST example (86.67% test) ✓, OpenMP ✓,
      conv2d/pool2d ✓, QCNN (MNIST/CIFAR/rotation) ✓, benchmark suite ✓.
- [ ] **Phase 5** — attention/transformer, model IO, Python bindings, publish.

## License

MIT — see [LICENSE](LICENSE).