#ifndef SCENE_SPHERE_H_
#define SCENE_SPHERE_H_

#include <cuda_runtime.h>
#include "util/surface.h"

typedef struct Sphere {
    float3 position;
    float radius;
    Surface surface;
} Sphere;

#endif // SCENE_SPHERE_H_
