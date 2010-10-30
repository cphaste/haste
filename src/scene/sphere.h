#ifndef SCENE_SPHERE_H_
#define SCENE_SPHERE_H_

#include "util/vectors.h"
#include "util/surface.h"

struct Sphere {
    float3 position;
    float radius;
    Surface surface;
};

typedef struct Sphere Sphere;

#endif // SCENE_SPHERE_H_
