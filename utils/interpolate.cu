#include "cuda_utils.h"
#include "interpolate.h"

__global__ void knn_kernel(int b, int n, int m, int k,
                           const float *__restrict__ unknown,
                           const float *__restrict__ known,
                           int *__restrict__ idxs,
                           float *__restrict__ dists,
                           float *__restrict__ temp) {
  const int batch_index = blockIdx.x;
  unknown += batch_index * n * 3;
  known += batch_index * m * 3;
  idxs += batch_index * n * k;
  dists += batch_index * n * k;
  temp += batch_index * n * m;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float ux = unknown[i * 3 + 0];
    float uy = unknown[i * 3 + 1];
    float uz = unknown[i * 3 + 2];
    float *d;
    d = temp + i * m;
    for (int j = 0; j < m; ++j) {
      float x = known[j * 3 + 0];
      float y = known[j * 3 + 1];
      float z = known[j * 3 + 2];
      d[j] = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
    }
    for (int l = 0; l < k; ++l) {
      int besti = -1;
      float best_dist = 1e5;
      for (int j = 0; j < m; ++j) {
        if(d[j] < best_dist) {
          besti = j;
          best_dist = d[j];
        }
      }
      d[besti] = 1e6;
      idxs[i * k + l] = besti;
      dists[i * k + l] = best_dist;
    }
  }
}

// input: unknown(b, n, 3), known(b, m, 3)
// output: idxs(b, n, k), dists(b, n, k)
void knn_kernel_wrapper(int b, int n, int m, int k, const float *unknown,
                        const float *known, int *idxs, float *dists, float *temp) {
  cudaError_t err;
  knn_kernel<<<b, opt_n_threads(n)>>>(b, n, m, k, unknown, known, idxs, dists, temp);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void interpolate_kernel(int b, int c, int m, int n, int k,
                                   const float *__restrict__ features,
                                   const int *__restrict__ idxs,
                                   const float *__restrict__ weights,
                                   float *__restrict__ out) {
  const int batch_index = blockIdx.x;
  features += batch_index * c * m;
  idxs += batch_index * n * k;
  weights += batch_index * n * k;
  out += batch_index * c * n;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      float sum = 0.;
      for (int l = 0; l < k; ++ l) {
        int id = idxs[j * k + l];
        sum += weights[j * k + l] * features[i * m + id];
      }
      out[i * blockDim.x + j] = sum;
    }
  }
}

// input: features(b, c, m), idxs(b, n, k), weights(b, n, k)
// output: out(b, c, n)
void interpolate_kernel_wrapper(int b, int c, int m, int n, int k,
                                const float *features, const int *idxs, const float *weights,
                                float *out) {
  cudaError_t err;
  interpolate_kernel<<<b, opt_block_config(n, c)>>>(b, c, m, n, k, features, idxs, weights, out);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void interpolate_grad_kernel(int b, int c, int m, int n, int k,
                                        const float *__restrict__ grad_out,
                                        const int *__restrict__ idxs,
                                        const float *__restrict__ weights,
                                        float *__restrict__ grad_points) {
  const int batch_index = blockIdx.x;
  grad_out += batch_index * c * n;
  idxs += batch_index * n * k;
  weights += batch_index * n * k;
  grad_points += batch_index * c * m;
                                        
  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      for (int l = 0; l < k; ++l) {
        int id = idxs[j * k + l];
        atomicAdd(grad_points + i * m + id, grad_out[i * n + j] * weights[j * k + l]);
      }
    }
  }
}

// input: grad_out(b, c, n), idxs(b, n, k), weights(b, n, k)
// output: grad_points(b, c, m)
void interpolate_grad_kernel_wrapper(int b, int c, int m, int n, int k,
                                     const float *grad_out, const int *idxs, const float *weights,
                                     float *grad_points) {
  cudaError_t err;
  interpolate_grad_kernel<<<b, opt_block_config(n, c)>>>(b, c, m, n, k, grad_out, idxs, weights, grad_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
