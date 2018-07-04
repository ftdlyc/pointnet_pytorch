#ifndef INTERPOLATE_H
#define INTERPOLATE_H

void knn_kernel_wrapper(int b, int n, int m, int k,
                        const float *unknown, const float *known,
                        int *idxs, float *dists, float *temp);

void interpolate_kernel_wrapper(int b, int c, int m, int n, int k,
                                const float *features, const int *idxs, const float *weights,
                                float *out);

void interpolate_grad_kernel_wrapper(int b, int c, int m, int n, int k,
                                     const float *grad_out, const int *idxs, const float *weights,
                                     float *grad_points);

#endif