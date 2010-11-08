#ifndef UTIL_RAY_H_
#define UTIL_RAY_H_

#include <stdint.h>
#include <cuda_runtime.h>

typedef struct Ray {
    float3 origin; // origin position of the ray
    float3 direction; // unit vector of the direction of travel
    float contrib; // contribution factor of this ray to its layer
    uint64_t layer; // layer this ray contributes to
    ushort2 pixel; // pixel this ray contributes to
    bool unibounce; // whether this ray should continue to bounce (spawn more rays)
} Ray;

#endif // UTIL_RAY_H_
