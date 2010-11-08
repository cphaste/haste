#ifndef UTIL_RENDER_H_
#define UTIL_RENDER_H_

#include <stdint.h>
#include <cuda_runtime.h>

typedef struct Render {
    ushort2 size; // size of the final render in pixels
    uint64_t max_bounces; // maxmimum number of bounces rays can make (recursion depth limit)
} Render;

#endif // UTIL_RENDER_H_
