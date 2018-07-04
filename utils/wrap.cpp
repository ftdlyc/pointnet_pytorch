#include <vector>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "group_points.h"
#include "interpolate.h"
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

at::Tensor farthest_point_sampling_wrapper(at::Tensor points_tensor, int m) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT_TYPE(points_tensor, at::ScalarType::Float);
  int b = points_tensor.size(0);
  int n = points_tensor.size(1);
  at::Tensor idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, m});
  at::Tensor temp_tensor = at::zeros(torch::CUDA(at::kFloat), {b, n}).fill_(1e6);
  const float *points = points_tensor.data<float>();
  float *temp = temp_tensor.data<float>();
  int *idxs = idxs_tensor.data<int>();

  farthest_point_sampling_kernel_wrapper(b, n, m, points, temp, idxs);
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

at::Tensor ball_query_wrapper(at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, float radius, int k) {
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

  ball_query_kernel_wrapper(b, n, m, radius, k, xyz, new_xyz, group_idxs);
  return group_idxs_tensor;
}

std::vector<at::Tensor> knn_wrapper(at::Tensor unknown_tensor, at::Tensor known_tensor, int k) {
  CHECK_INPUT(unknown_tensor);
  CHECK_INPUT_TYPE(unknown_tensor, at::ScalarType::Float);
  CHECK_INPUT(known_tensor);
  CHECK_INPUT_TYPE(known_tensor, at::ScalarType::Float);
  int b = unknown_tensor.size(0);
  int n = unknown_tensor.size(1);
  int m = known_tensor.size(1);
  AT_ASSERT(m > k, "k must smaller than m")
  at::Tensor idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, n, k});
  at::Tensor dists_tensor = at::zeros(torch::CUDA(at::kFloat), {b, n, k});
  at::Tensor temp_tensor = at::zeros(torch::CUDA(at::kFloat), {b, n, m});
  const float *unknown = unknown_tensor.data<float>();
  const float *known = known_tensor.data<float>();
  int *idxs = idxs_tensor.data<int>();
  float *dists = dists_tensor.data<float>();
  float *temp = temp_tensor.data<float>();

  knn_kernel_wrapper(b, n, m, k, unknown, known, idxs, dists, temp);
  return {idxs_tensor, dists_tensor};
}

at::Tensor interpolate_wrapper(at::Tensor features_tensor, at::Tensor idxs_tensor, at::Tensor weights_tensor) {
  CHECK_INPUT(features_tensor);
  CHECK_INPUT_TYPE(features_tensor, at::ScalarType::Float);
  CHECK_INPUT(idxs_tensor);
  CHECK_INPUT_TYPE(idxs_tensor, at::ScalarType::Int);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  int b = features_tensor.size(0);
  int c = features_tensor.size(1);
  int m = features_tensor.size(2);
  int n = idxs_tensor.size(1);
  int k = idxs_tensor.size(2);
  at::Tensor out_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, n});
  const float *features = features_tensor.data<float>();
  const int *idxs = idxs_tensor.data<int>();
  const float *weights = weights_tensor.data<float>();
  float *out = out_tensor.data<float>();

  interpolate_kernel_wrapper(b, c, m, n, k, features, idxs, weights, out);
  return out_tensor;
}

at::Tensor interpolate_grad_wrapper(at::Tensor grad_out_Tensor, at::Tensor idxs_tensor, at::Tensor weights_tensor, int m) {
  CHECK_INPUT(grad_out_Tensor);
  CHECK_INPUT_TYPE(grad_out_Tensor, at::ScalarType::Float);
  CHECK_INPUT(idxs_tensor);
  CHECK_INPUT_TYPE(idxs_tensor, at::ScalarType::Int);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  int b = grad_out_Tensor.size(0);
  int c = grad_out_Tensor.size(1);
  int n = idxs_tensor.size(1);
  int k = idxs_tensor.size(2);
  at::Tensor grad_points_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, m});
  const float *grad_out = grad_out_Tensor.data<float>();
  const int *idxs = idxs_tensor.data<int>();
  const float *weights = weights_tensor.data<float>();
  float *grad_points = grad_points_tensor.data<float>();

  interpolate_grad_kernel_wrapper(b, c, m, n, k, grad_out, idxs, weights, grad_points);
  return grad_points_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling_wrapper", &farthest_point_sampling_wrapper, "farthest point sampling");
    m.def("gather_points_wrapper", &gather_points_wrapper, "gather points");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper, "gather points grad");
    m.def("ball_query_wrapper", &ball_query_wrapper, "ball query");
    m.def("group_points_wrapper", &group_points_wrapper, "group points");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper, "group points grad");
    m.def("knn_wrapper", &knn_wrapper, "knn");
    m.def("interpolate_wrapper", &interpolate_wrapper, "interpolate");
    m.def("interpolate_grad_wrapper", &interpolate_grad_wrapper, "interpolate grad");
}
