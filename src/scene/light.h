#ifndef SCENE_LIGHT_H_
#define SCENE_LIGHT_H_

#include <cuda_runtime.h>

typedef struct Light {
    float3 position;
    float3 color;
} Light;

#endif // SCENE_LIGHT_H_
