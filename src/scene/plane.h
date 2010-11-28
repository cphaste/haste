#ifndef SCENE_PLANE_H_
#define SCENE_PLANE_H_

#include <stdint.h>
#include <cuda_runtime.h>

typedef struct Plane {
    float3 normal;
    float distance;
    uint64_t material;
} Plane;

#endif // SCENE_PLANE_H_
