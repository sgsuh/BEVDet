/*
Create: 2026.05.30
Author: SG.SUH
*/

#include <cstdio>
#include <cstdlib>
#include "trt_bev_pool_kernel.h"

__global__ void bevPoolV2Kernel(int c,
                                int n_intervals,
                                const float* __restrict__ depth,
                                const float* __restrict__ feat,
                                const int* __restrict__ ranks_depth,
                                const int* __restrict__ ranks_feat,
                                const int* __restrict__ ranks_bev,
                                const int* __restrict__ interval_starts,
                                const int* __restrict__ interval_lengths,
                                float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;

    if(index >= n_intervals) {
        return;
    }

    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    float psum = 0;
    const float* cur_depth;
    const float* cur_feat;

    for(int i = 0; i < interval_length; i++) {
        cur_depth = depth + ranks_depth[interval_start + i];
        cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
        psum += *cur_feat * *cur_depth;
    }

    const int* cur_rank = ranks_bev + interval_start;
    float* cur_out = out + *cur_rank * c + cur_c;
    *cur_out = psum;
}

__global__ void bevPoolV2SetZeroKernel(int n_points, float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n_points) {
        return;
    }

    float* cur_out = out + idx;
    *cur_out = 0.0;
}

void bevPoolV2(int c,
                int n_intervals,
                const float* depth,
                const float* feat,
                const int* ranks_depth,
                const int* ranks_feat,
                const int* ranks_bev,
                const int* interval_starts,
                const int* interval_lengths,
                float* out,
                cudaStream_t stream) {
    bevPoolV2Kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256, 0, stream>>>(c, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, out);
}

void bevPoolV2SetZero(int n_points, float* out) {
    bevPoolV2SetZeroKernel<<<(int)ceil(((double)n_points / 256)), 256>>>(n_points, out);
}