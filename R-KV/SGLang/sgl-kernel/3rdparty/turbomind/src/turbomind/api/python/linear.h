// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/tensor.h"
#include <cuda_runtime.h>
#include <istream>
#include <memory>
#include <ostream>

namespace turbomind {

enum class WeightType : int
{
    kFP32,
    kFP16,
    kFP8,  // not supported yet
    kBF16,
    kINT8,
    kINT4
};

class Linear {
public:
    Linear(size_t input_dims, size_t output_dims, int w_bit, int group_size);
    void post_init(std::shared_ptr<Tensor> qweight, const Tensor& scales, const Tensor& qzeros, bool simt);
    void forward(const Tensor& in, Tensor& out, cudaStream_t stream = nullptr);
    ~Linear() {}

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
};  // namespace turbomind
