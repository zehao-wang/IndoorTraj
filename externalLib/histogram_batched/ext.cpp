#include <torch/extension.h>
#include "histogram_batched.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("histogram_batched", &histogram2d_batched);
  m.def("histogram_batched_c", &histogram2d_c_batched);
}
