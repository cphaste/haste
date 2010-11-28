#ifndef SCENE_TRIANGLE_H_
#define SCENE_TRIANGLE_H_

#include <stdint.h>
#include <cuda_runtime.h>

typedef struct Triangle {
    float3 vertex1;
    float3 vertex2;
    float3 vertex3;
    float3 normal1;
    float3 normal2;
    float3 normal3;
    uint64_t material;
} Triangle;

#endif // SCENE_TRIANGLE_H_
