#ifndef DEVICE_RAYTRACE_CU_H_
#define DEVICE_RAYTRACE_CU_H_

#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "util/traceparams.h"
#include "util/ray_packet.h"
#include "util/material.h"
#include "util/render.h"
#include "scene/objtypes.h"
#include "scene/sphere.h"
#include "scene/triangle.h"

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
	__device__ float3 operator/(const float3 &lhs, const float &wgt);
	__device__ float3 operator*(const float3 &lhs, const float &wgt);
    __device__ float dot(const float3 &lhs, const float3 &rhs);
    __device__ float3 cross(const float3 &lhs, const float3 &rhs);
    __device__ float length(const float3 &v);
    __device__ float distance(const float3 &a, const float3 &b);
    __device__ float3 normalize(const float3 &v);
    __device__ float3 evaluate(Ray *ray, float t);
    __device__ float triarea(const float3 &a, const float3 &b, const float3 &c);
    
    // randomness helpers
    __device__ curandState GetRandState();

    // normal functions
    __device__ float3 Normal(Sphere *sphere, const float3 &point);
    __device__ float3 Normal(Triangle *triangle, const float3 &point);
    __device__ float3 Normal(Triangle *triangle);
    __device__ float3 Normal(Intersection *obj, const float3 &point);

    // intersection functions
    __device__ float Intersect(Ray *ray, Sphere *sphere);
    __device__ float Intersect(Ray *ray, Triangle *triangle);
    __device__ bool Intersect(Ray *ray, Intersection *obj);
    __device__ Intersection NearestObj(Ray *ray);

    // accessor functions
    __device__ float3 GetLayerBuffer(ushort2 pixel, uint64_t layer);
    __device__ void SetLayerBuffer(ushort2 pixel, uint64_t layer, float3 color);
    __device__ void BlendWithLayerBuffer(ushort2 pixel, uint64_t layer, float3 color);

    // shading functions
    __device__ Material* GetMaterial(Intersection *obj);
    __device__ float3 GetLightColor(LightObject *light);
    __device__ float3 GetRandomLightPosition(curandState *rand_state, LightObject *light);
    __device__ void DirectShading(Ray *ray, Intersection *obj);

    // kernels
    __global__ void InitRandomness(uint64_t seed, curandState *rand_states);
    __global__ void RayTrace(RayPacket packet);
}

#endif // DEVICE_RAYTRACE_CU_H_
