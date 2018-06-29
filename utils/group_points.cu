#include "cuda_utils.h"
#include "group_points.h"

__global__ void group_points_kernel(int b, int c, int n, int m, int k,
                                    const float *__restrict__ points,
                                    const int *__restrict__ group_idxs,
                                    float *__restrict__ out) {
  const int batch_index = blockIdx.x;
  points += batch_index * c * n;
  group_idxs += batch_index * m * k;
  out += batch_index * c * m * k;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      for (int l = 0; l < k; ++l) {
        out[(i * m + j) * k + l] = points[i * n + group_idxs[j * k + l]];
      }
    }
  }
}

// input: points(b, c, n) group_idxs(b, m, k)
// output: out(b, c, m, k)
void group_points_kernel_wrapper(int b, int c, int n, int m, int k,
                                 const float *points, const int *group_idxs,
                                 float *out) {
  cudaError_t err;
  group_points_kernel<<<b, opt_block_config(m, c)>>>(
      b, c, n, m, k, points, group_idxs, out);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void group_points_grad_kernel(int b, int c, int n, int m, int k,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ group_idxs, 
                                         float *__restrict__ grad_points) {
  const int batch_index = blockIdx.x;
  grad_out += batch_index * c * m * k;
  group_idxs += batch_index * m * k;
  grad_points += batch_index * c * n;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      for (int l = 0; l < k; ++l) {
        atomicAdd(grad_points + i * n + group_idxs[j * k + l], grad_out[(i * m + j) * k + l]);
      }
    }
  }
}

// input: grad_out(b, c, m, k), group_idxs(b, m, k)
// output: grad_points(b, c, n)
void group_points_grad_kernel_wrapper(int b, int c, int n, int m, int k,
                                      const float *grad_out,
                                      const int *group_idxs,
                                      float *grad_points) {
  cudaError_t err;
  group_points_grad_kernel<<<b, opt_block_config(m, c)>>>(
      b, c, n, m, k, grad_out, group_idxs, grad_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void ball_query_kernel(int b, int n, int m, float radius2,
                                  int k, const float *__restrict__ xyz,
                                  const float *__restrict__ new_xyz,
                                  int *__restrict__ group_idxs) {
  xyz += blockIdx.x * n * 3;
  new_xyz += blockIdx.x * m * 3;
  group_idxs += blockIdx.x * k * m;

  for (int i = threadIdx.x; i < m; i += blockDim.x) {
    float new_x = new_xyz[i * 3 + 0];
    float new_y = new_xyz[i * 3 + 1];
    float new_z = new_xyz[i * 3 + 2];
    int cnt = 0;
    for (int j = 0; j < n; ++j) {
      float x = xyz[j * 3 + 0];
      float y = xyz[j * 3 + 1];
      float z = xyz[j * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < k; ++l) {
            group_idxs[i * k + l] = j;
          }
        }
        group_idxs[i * k + cnt] = j;
        ++cnt;
      }
      if (cnt >= k) {
        break;
      }
    }
  }
}

// input: xyz(b, n, 3), new_xyz(b, m, 3)
// output: idx(b, m, k)
void ball_query_kernel_wraper(int b, int n, int m, float radius, int k,
                              const float *xyz, const float *new_xyz,
                              int *group_idxs) {
  cudaError_t err;
  float radius2 = radius * radius;
  ball_query_kernel<<<b, opt_n_threads(m)>>>(
      b, n, m, radius2, k, xyz, new_xyz, group_idxs);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
