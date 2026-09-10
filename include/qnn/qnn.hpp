#ifndef QNN_QNN_HPP
#define QNN_QNN_HPP

// Master include for the quaternion-nn library. Include this single header to
// get the whole public API: core math, functional ops, layers, models,
// optimizers, losses and data loaders.

#include "qnn/core/quaternion.hpp"
#include "qnn/core/quaternion_matrix.hpp"
#include "qnn/core/quaternion_vector.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

#include "qnn/data/mnist.hpp"

#include "qnn/functional/convolution.hpp"
#include "qnn/functional/flatten.hpp"
#include "qnn/functional/matmul.hpp"
#include "qnn/functional/pool.hpp"
#include "qnn/functional/relu.hpp"
#include "qnn/functional/sigmoid.hpp"
#include "qnn/functional/softmax.hpp"
#include "qnn/functional/tanh.hpp"

#include "qnn/loss/cross_entropy.hpp"
#include "qnn/loss/mse.hpp"

#include "qnn/nn/init.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/nn/parameter.hpp"
#include "qnn/nn/sequential.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/conv2d.hpp"
#include "qnn/nn/layers/flatten.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/layers/pool2d.hpp"
#include "qnn/nn/models/qcnn.hpp"

#include "qnn/optim/adam.hpp"
#include "qnn/optim/optimizer.hpp"
#include "qnn/optim/sgd.hpp"

#endif  // QNN_QNN_HPP