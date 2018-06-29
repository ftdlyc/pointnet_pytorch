#ifndef GROUP_POINTS_H
#define GROUP_POINTS_H

void group_points_kernel_wrapper(int b, int c, int n, int m, int k,
                                 const float *points, const int *group_idxs,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int m, int k,
                                      const float *grad_out, const int *group_idxs,
                                      float *grad_points);

void ball_query_kernel_wraper(int b, int n, int m, float radius, int k,
                              const float *xyz, const float *new_xyz,
                              int *group_idxs);

#endif