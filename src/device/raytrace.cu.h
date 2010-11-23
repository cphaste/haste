#ifndef DEVICE_RAYTRACE_CU_H_
#define DEVICE_RAYTRACE_CU_H_

#include <stdint.h>
#include <cuda_runtime.h>

#include "util/traceparams.h"
#include "util/surface.h"
#include "util/render.h"
#include "scene/objtypes.h"
#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/plane.h"

typedef struct Intersection {
    ObjType type;
    void *ptr;
    float t;
} Intersection;

namespace device {
    // math functions
    __device__ float3 operator+(const float3 &lhs, const float3 &rhs);
    __device__ float3 operator-(const float3 &lhs, const float3 &rhs);
    __device__ float3 operator*(const float3 &lhs, const float3 &rhs);
    __device__ float3 operator/(const float3 &lhs, const float3 &rhs);
    __device__ float dot(const float3 &lhs, const float3 &rhs);
    __device__ float length(const float3 &v);
    __device__ float3 normalize(const float3 &v);
    __device__ float3 evaluate(Ray *ray, float t);

    // normal functions
    __device__ float3 Normal(Sphere *sphere, const float3 &point);
    __device__ float3 Normal(Plane *plane, const float3 &point);
    __device__ float3 Normal(Intersection *obj, const float3 &point);

    // intersection functions
    __device__ float Intersect(Ray *ray, Sphere *sphere);
    __device__ float Intersect(Ray *ray, Plane *plane);
    __device__ bool Intersect(Ray *ray, Intersection *obj);
    __device__ Intersection NearestObj(Ray *ray, TraceParams *params);

    // accessor functions
    __device__ float3 GetLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer);
    __device__ void SetLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer, float3 color);
    __device__ void BlendWithLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer, float3 color);
    __device__ Surface* GetSurface(Intersection *obj);
    __device__ float3 GetLightColor(TraceParams *params, MetaObject *obj);
    __device__ float3 GetRandomLightPosition(TraceParams *params, MetaObject *obj);

    // shading functions
    __device__ void DirectShading(TraceParams *params, Ray *ray, Intersection *obj);

    // kernels
    __global__ void RayTrace(TraceParams *params);
}

#endif // DEVICE_RAYTRACE_CU_H_
