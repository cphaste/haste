#ifndef UTIL_RENDER_H_
#define UTIL_RENDER_H_

#include <stdint.h>
#include <cuda_runtime.h>

typedef struct Render {
    ushort2 size; // size of the final render in pixels
    uint32_t max_bounces; // maxmimum number of bounces rays can make (recursion depth limit)
    uint32_t antialiasing; // antialiasing grid size (if this is n, it will stochastically sample a n x n grid)
    uint32_t direct_samples; // number of samples to take per light for direct lighting
    uint32_t indirect_samples; // number of samples to take for indirect lighting
} Render;

#endif // UTIL_RENDER_H_
