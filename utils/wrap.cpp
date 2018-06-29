#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "group_points.h"
#include "sampling.h"

at::Tensor gather_points_wrapper(at::Tensor points_tensor, at::Tensor idxs_tensor) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT_TYPE(points_tensor, at::ScalarType::Float);
  CHECK_INPUT(idxs_tensor);
  CHECK_INPUT_TYPE(idxs_tensor, at::ScalarType::Int);
  int b = points_tensor.size(0);
  int c = points_tensor.size(1);
  int n = points_tensor.size(2);
  int m = idxs_tensor.size(1);
  at::Tensor out_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, m});
  const float *points = points_tensor.data<float>();
  const int *idxs = idxs_tensor.data<int>();
  float *out = out_tensor.data<float>();

  gather_points_kernel_wrapper(b, c, n, m, points, idxs, out);
  return out_tensor;
}

at::Tensor gather_points_grad_wrapper(at::Tensor grad_out_Tensor, at::Tensor idxs_tensor, int n) {
  CHECK_INPUT(grad_out_Tensor);
  CHECK_INPUT_TYPE(grad_out_Tensor, at::ScalarType::Float);
  CHECK_INPUT(idxs_tensor);
  CHECK_INPUT_TYPE(idxs_tensor, at::ScalarType::Int);
  int b = grad_out_Tensor.size(0);
  int c = grad_out_Tensor.size(1);
  int m = grad_out_Tensor.size(2);
  at::Tensor grad_points_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, n});
  const float *grad_out = grad_out_Tensor.data<float>();
  const int *idxs = idxs_tensor.data<int>();
  float *grad_points = grad_points_tensor.data<float>();

  gather_points_grad_kernel_wrapper(b, c, n, m, grad_out, idxs, grad_points);
  return grad_points_tensor;
}

at::Tensor farthest_point_sampling_wraper(at::Tensor points_tensor, int m) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT_TYPE(points_tensor, at::ScalarType::Float);
  int b = points_tensor.size(0);
  int n = points_tensor.size(1);
  at::Tensor idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, m});
  at::Tensor temp_tensor = at::zeros(torch::CUDA(at::kFloat), {b, n}).fill_(1e6);
  const float *points = points_tensor.data<float>();
  float *temp = temp_tensor.data<float>();
  int *idxs = idxs_tensor.data<int>();

  farthest_point_sampling_kernel_wraper(b, n, m, points, temp, idxs);
  return idxs_tensor;
}

at::Tensor group_points_wrapper(at::Tensor points_tensor, at::Tensor group_idxs_tensor) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT_TYPE(points_tensor, at::ScalarType::Float);
  CHECK_INPUT(group_idxs_tensor);
  CHECK_INPUT_TYPE(group_idxs_tensor, at::ScalarType::Int);
  int b = points_tensor.size(0);
  int c = points_tensor.size(1);
  int n = points_tensor.size(2);
  int m = group_idxs_tensor.size(1);
  int k = group_idxs_tensor.size(2);
  at::Tensor out_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, m, k});
  const float *points = points_tensor.data<float>();
  const int *group_idxs = group_idxs_tensor.data<int>();
  float *out = out_tensor.data<float>();

  group_points_kernel_wrapper(b, c, n, m, k, points, group_idxs, out);
  return out_tensor;
}

at::Tensor group_points_grad_wrapper(at::Tensor grad_out_Tensor, at::Tensor group_idxs_tensor, int n) {
  CHECK_INPUT(grad_out_Tensor);
  CHECK_INPUT_TYPE(grad_out_Tensor, at::ScalarType::Float);
  CHECK_INPUT(group_idxs_tensor);
  CHECK_INPUT_TYPE(group_idxs_tensor, at::ScalarType::Int);
  int b = grad_out_Tensor.size(0);
  int c = grad_out_Tensor.size(1);
  int m = grad_out_Tensor.size(2);
  int k = grad_out_Tensor.size(3);
  at::Tensor grad_points_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, n});
  const float *grad_out = grad_out_Tensor.data<float>();
  const int *group_idxs = group_idxs_tensor.data<int>();
  float *grad_points = grad_points_tensor.data<float>();

  group_points_grad_kernel_wrapper(b, c, n, m, k, grad_out, group_idxs, grad_points);
  return grad_points_tensor;
}

at::Tensor ball_query_wraper(at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, float radius, int k) {
  CHECK_INPUT(xyz_tensor);
  CHECK_INPUT_TYPE(xyz_tensor, at::ScalarType::Float);
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT_TYPE(new_xyz_tensor, at::ScalarType::Float);
  int b = new_xyz_tensor.size(0);
  int m = new_xyz_tensor.size(1);
  int n = xyz_tensor.size(1);
  at::Tensor group_idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, m, k});
  const float *xyz = xyz_tensor.data<float>();
  const float *new_xyz = new_xyz_tensor.data<float>();
  int *group_idxs = group_idxs_tensor.data<int>();

  ball_query_kernel_wraper(b, n, m, radius, k, xyz, new_xyz, group_idxs);
  return group_idxs_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling_wraper", &farthest_point_sampling_wraper, "farthest point sampling");
    m.def("gather_points_wrapper", &gather_points_wrapper, "gather points");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper, "gather points grad");
    m.def("ball_query_wraper", &ball_query_wraper, "ball query");
    m.def("group_points_wrapper", &group_points_wrapper, "group points");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper, "group points grad");
}
