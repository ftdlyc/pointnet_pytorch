#ifndef SAMPLING_H
#define SAMPLING_H

void gather_points_kernel_wrapper(int b, int c, int n, int m,
                                  const float *points, const int *idxs,
                                  float *out);

void gather_points_grad_kernel_wrapper(int b, int c, int n, int m,
                                       const float *grad_out, const int *idxs,
                                       float *grad_points);

void farthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                           const float *points, float *temp,
                                           int *idxs);

#endif