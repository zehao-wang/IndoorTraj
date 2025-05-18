#include "histogram_batched.h"

#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


namespace cg = cooperative_groups;


__global__ void histogram2d_batched_kernel(float* x, float* y, int* mask,
                                           int samples,
                                           int bins, float x_min, float x_max,
                                           float y_min, float y_max,
                                           int batches, float *hist_out)
{
    int idx = cg::this_grid().thread_rank();
    if (idx > batches - 1) return;

    float bin_size_x = (x_max-x_min)/(float)(2*bins);
    float bin_size_y = (y_max-y_min)/(float)bins;

    for (int i=0; i<samples; i++) {
        if (mask[idx*samples + i]==0) continue;

        int x_idx_hist = floor(x[idx*samples + i]/bin_size_x);
        int y_idx_hist = floor(y[idx*samples + i]/bin_size_y);
        hist_out[idx*bins*2*bins + y_idx_hist*2*bins + x_idx_hist] += 1;
    }
}

torch::Tensor histogram2d_batched(const torch::Tensor& x,
                                  const torch::Tensor& y,
                                  const torch::Tensor& mask,
                                  const int bins,
                                  const float x_min,
                                  const float x_max,
                                  const float y_min,
                                  const float y_max)
{
  const int batches = x.size(0);
  const int samples = x.size(1);

  auto float_opts = x.options().dtype(torch::kFloat32);
  torch::Tensor hist_out = torch::full({batches, bins, 2*bins}, 0.0, float_opts);

  histogram2d_batched_kernel<<<(batches+255)/256, 256>>>(x.contiguous().data<float>(), y.contiguous().data<float>(),
                                                         mask.contiguous().data<int>(),
                                                         samples, bins, x_min, x_max, y_min, y_max,
                                                         batches, hist_out.contiguous().data<float>());

  return hist_out;
}


__global__ void histogram2d_c_batched_kernel(float* x, float* y, int* mask, int* cmask,
                                           int samples,
                                           int bins, int channels, float x_min, float x_max,
                                           float y_min, float y_max,
                                           int batches, float *hist_out)
{
    int idx = cg::this_grid().thread_rank();
    if (idx > batches - 1) return;

    float bin_size_x = (x_max-x_min)/(float)(2*bins);
    float bin_size_y = (y_max-y_min)/(float)bins;

    for (int i=0; i<samples; i++) {
        if (mask[idx*samples + i]==0) continue;

        int c = cmask[idx*samples + i];
        int x_idx_hist = floor(x[idx*samples + i]/bin_size_x);
        int y_idx_hist = floor(y[idx*samples + i]/bin_size_y);
        hist_out[idx*channels*bins*2*bins + c * bins * 2 * bins + y_idx_hist*2*bins + x_idx_hist] += 1;
    }
}

torch::Tensor histogram2d_c_batched(const torch::Tensor& x,
                                  const torch::Tensor& y,
                                  const torch::Tensor& mask,
                                  const torch::Tensor& cmask,
                                  const int bins,
                                  const int channels,
                                  const float x_min,
                                  const float x_max,
                                  const float y_min,
                                  const float y_max)
{
  const int batches = x.size(0);
  const int samples = x.size(1);

  auto float_opts = x.options().dtype(torch::kFloat32);
  torch::Tensor hist_out = torch::full({batches, channels, bins, 2*bins}, 0.0, float_opts);

  histogram2d_c_batched_kernel<<<(batches+255)/256, 256>>>(x.contiguous().data<float>(), y.contiguous().data<float>(),
                                                         mask.contiguous().data<int>(), cmask.contiguous().data<int>(),
                                                         samples, bins, channels, x_min, x_max, y_min, y_max,
                                                         batches, hist_out.contiguous().data<float>());

  return hist_out;
}
