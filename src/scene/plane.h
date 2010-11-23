#ifndef SCENE_PLANE_H_
#define SCENE_PLANE_H_

#include <cuda_runtime.h>
#include "util/surface.h"

typedef struct Plane {
    float3 normal;
    float distance;
    Surface surface;
} Plane;

#endif // SCENE_PLANE_H_
