#ifndef SCENE_SPHERE_H_
#define SCENE_SPHERE_H_

#include <stdint.h>
#include <cuda_runtime.h>

typedef struct Sphere {
    float3 position;
    float radius;
    uint64_t material;
} Sphere;

#endif // SCENE_SPHERE_H_
