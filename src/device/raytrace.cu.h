#ifndef DEVICE_RAYTRACE_CU_H_
#define DEVICE_RAYTRACE_CU_H_

#include <stdint.h>
#include <cuda_runtime.h>

#include "util/traceparams.h"
#include "scene/objtypes.h"
#include "scene/sphere.h"

typedef struct Intersection {
    ObjType type;
    void *ptr;
    float t;
} Intersection;

namespace device {
    // math functions
    __device__ float3 operator+(const float3 &lhs, const float3 &rhs);
    __device__ float3 operator-(const float3 &lhs, const float3 &rhs);
    __device__ float3 operator*(const float4 &lhs, const float3 &rhs);
    __device__ float3 operator/(const float4 &lhs, const float3 &rhs);
    __device__ float dot(const float3 &lhs, const float3 &rhs);

    // intersection functions
    __device__ float Intersect(Ray *ray, Sphere *sphere);
    __device__ bool Intersect(Ray *ray, Intersection *obj);
    __device__ Intersection NearestObj(Ray *ray, TraceParams *params);

    // accessor functions
    __device__ float3 GetLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer);
    __device__ void SetLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer, float3 color);

    // kernels
    __global__ void RayTrace(TraceParams *params);
}

#endif // DEVICE_RAYTRACE_CU_H_
