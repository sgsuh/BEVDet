/*
Create: 2026.05.29
Author: SG.SUH
*/

#pragma once

#include <cuda_runtime.h>
#include "common_cuda_helper.hpp"

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
                cudaStream_t stream);

void bevPoolV2SetZero(int n_points, float* out);