#include "qnn/qnn.hpp"

using qf = qnn::quaternion<float>;
using qnn::shape;
using qnn::tensor;

// ---------------------------------------------------------------------------
// 1) QCNN model from Config — 28x28 grayscale in, MNIST-like head of 10.
// ---------------------------------------------------------------------------
qnn::nn::models::qcnn<float> make_model() {
    qnn::nn::models::qcnn<float>::Config cfg;
    cfg.in_channels = 1;
    cfg.height = 28;
    cfg.width = 28;
    cfg.channels = {8, 16};        // conv width per block
    cfg.kernel = 3;
    cfg.padding = 1;
    cfg.stride = 1;
    cfg.pool_size = 2;
    cfg.pool_stride = 2;
    cfg.pool_type = qnn::functional::pool_type::max;
    cfg.activation = qnn::nn::layers::split_activation<float>::kind::tanh;
    cfg.out_features = 10;
    cfg.seed = 7;
    return qnn::nn::models::qcnn<float>(cfg);
}

// ---------------------------------------------------------------------------
// 2) Training loop — SGD bound to the whole model, images [N,1,H,W].
// ---------------------------------------------------------------------------
void train(qnn::nn::models::qcnn<float>& model,
           const tensor<qf>& x, const tensor<qf>& tgt) {
    qnn::optim::sgd<float> opt(0.01f);
    opt.add(model);                      // binds every parameter+grad once
    const float scale = 2.0f / static_cast<float>(x.dim(0) * tgt.dim(1) * 4);

    for (int ep = 0; ep < 10; ++ep) {
        opt.zero_grad();
        tensor<qf> y = model.forward(x);          // [N, out_features]
        float loss = qnn::loss::mse(y, tgt);

        tensor<qf> dy(y.shape());
        for (std::size_t i = 0; i < y.size(); ++i) dy[i] = (y[i] - tgt[i]) * scale;
        model.backward(dy);
        opt.step();

        std::printf("ep %d  loss %.4f\n", ep, loss);
    }
}

// ---------------------------------------------------------------------------
// 3) The same idea with a raw sequential — stack any Modules manually.
// ---------------------------------------------------------------------------
void manual() {
    qnn::nn::sequential<float> net;
    net.add<qnn::nn::layers::linear>(784, 128, true, 1);
    net.add<qnn::nn::layers::split_activation>(
        qnn::nn::layers::split_activation<float>::kind::relu);
    net.add<qnn::nn::layers::linear>(128, 10, true, 2);

    qnn::optim::sgd<float> opt(0.05f);
    opt.add(net);                        // sequential also exposes parameters()

    tensor<qf> x(shape{32, 784});        // already flattened images
    tensor<qf> t(shape{32, 10});
    // ... fill x, t, then the same zero_grad/forward/backward/step loop ...
}