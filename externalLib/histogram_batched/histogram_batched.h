#include <torch/extension.h>

torch::Tensor histogram2d_batched(const torch::Tensor& x,
                                  const torch::Tensor& y,
                                  const torch::Tensor& mask,
                                  const int bins,
                                  const float x_min,
                                  const float x_max,
                                  const float y_min,
                                  const float y_max);

torch::Tensor histogram2d_c_batched(const torch::Tensor& x,
                                  const torch::Tensor& y,
                                  const torch::Tensor& mask,
                                  const torch::Tensor& cmask,
                                  const int bins,
                                  const int channels,
                                  const float x_min,
                                  const float x_max,
                                  const float y_min,
                                  const float y_max);