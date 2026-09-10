import os

import numpy as np

# ============================================================================
# Shared rotation-estimation data for both sides of the benchmark.
#
# Sample: input = image of the basis (e1,e2,e3) under a random rotation R(q),
#         9 floats ([3 vectors] x [3 coords]);
#         target = the rotation as a unit quaternion q (4 floats).
# Uniform on SO(3). Deterministic (seed 7). The same function is embedded in
# benchmarks/rotation_torch.py so Colab can regenerate byte-identical bins.
# ============================================================================


def sample(n, seed=7):
    rng = np.random.default_rng(seed)
    u1 = rng.random(n).astype(np.float32)
    u2 = rng.random(n).astype(np.float32)
    u3 = rng.random(n).astype(np.float32)

    # uniform quaternion on S^3
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    q = np.stack([w, x, y, z], axis=1)                # [n,4]

    # rotation matrices (conventional R for unit quat (w,x,y,z)) -> [n,3,3]
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
    R = np.stack([row0, row1, row2], axis=1)          # [n,3,3]
    # v_j = R(q) e_j  == column j of R
    v0 = R[:, :, 0]  # [n,3], image of e1
    v1 = R[:, :, 1]  # image of e2
    v2 = R[:, :, 2]  # image of e3
    inputs = np.concatenate([v0, v1, v2], axis=1).astype(np.float32)  # [n,9]
    return inputs, q


def store(path, inputs, targets):
    n = inputs.shape[0]
    with open(path, "wb") as f:
        f.write(np.asarray(n, dtype=np.int32).tobytes())
        f.write(np.asarray(inputs, dtype=np.float32).tobytes())
        f.write(np.asarray(targets, dtype=np.float32).tobytes())
    print(f"{path}: {n} samples (9+4 float32 each, {inputs.nbytes / 1e6:.1f} MB)")


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "data", "rotation")
    os.makedirs(root, exist_ok=True)
    train_in, train_q = sample(12000)
    test_in, test_q = sample(2000)
    store(os.path.join(root, "train.bin"), train_in, train_q)
    store(os.path.join(root, "test.bin"), test_in, test_q)


if __name__ == "__main__":
    main()