#include "raytrace.cu.h"

namespace device {
    __device__ const float EPSILON = 0.0001f;
}

// ===== math functions =====

__device__ float3 device::operator+(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x + rhs.x;
    r.y = lhs.y + rhs.y;
    r.z = lhs.z + rhs.z;
    return r;
}

__device__ float3 device::operator-(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x - rhs.x;
    r.y = lhs.y - rhs.y;
    r.z = lhs.z - rhs.z;
    return r;
}

__device__ float3 device::operator*(const float4 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x * rhs.x;
    r.y = lhs.y * rhs.y;
    r.z = lhs.z * rhs.z;
    return r;
}

__device__ float3 device::operator/(const float4 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x / rhs.x;
    r.y = lhs.y / rhs.y;
    r.z = lhs.z / rhs.z;
    return r;
}

__device__ float device::dot(const float3 &lhs, const float3 &rhs) {
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

// ===== intersection functions =====

__device__ float device::Intersect(Ray *ray, Sphere *sphere) {
    // calculate quadratic components
    float a = dot(ray->direction, ray->direction);
    float b = dot(ray->direction, (ray->origin - sphere->position)) * 2.0f;
    float c = dot((ray->origin - sphere->position), (ray->origin - sphere->position)) - (sphere->radius * sphere->radius);

    float discr = (b * b) - (4.0f * a * c);
    if (discr > 0.0f) {
        // two intersections, return the closest positive t
        float t1 = ((-1.0f * b) + sqrtf(discr)) / (2.0f * a);
        float t2 = ((-1.0f * b) - sqrtf(discr)) / (2.0f * a);

        if (t1 > EPSILON && t2 > EPSILON) {
            return fminf(t1, t2);
        } else if (t1 < -EPSILON && t2 < -EPSILON) {
            return fmaxf(t1, t2);
        } else {
            return (fabs(t1) < EPSILON) ? t2 : t1;
        }
    } else if (discr == 0.0f) {
        // barely grazes the edge, only a single intersection
        return (-1.0f * b) / (2.0f * a);
    }

    // does not intersect the sphere
    return -1.0f;
}

__device__ bool device::Intersect(Ray *ray, Intersection *obj) {
    // TODO: transform the ray by the inverse transformation matrix

    switch (obj->type) {
        case SPHERE:
            obj->t = Intersect(ray, (Sphere *)obj->ptr);
            if (obj->t > EPSILON) return true;
            break;
    }

    return false;
}

__device__ Intersection device::NearestObj(Ray *ray, TraceParams *params) {
    Intersection closest = {SPHERE, NULL, -1.0f};
 
    // check all objects for intersections
    for (uint64_t i = 0; i < params->num_objs; i++) {
        Intersection obj;
        obj.type = params->meta_chunk[i].type;
        obj.ptr = (void *) ((uint64_t) params->obj_chunk + params->meta_chunk[i].offset);
        if (Intersect(ray, &obj)) {
            if (closest.t < 0.0f) {
                closest = obj;
            } else {
                if (obj.t < closest.t) {
                    closest = obj;
                }
            }
        }
    }

    return closest;
}

// ===== accessor functions =====

__device__ float3 device::GetLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer) {
    uint64_t layer_offset = sizeof(float3) * params->size.x * params->size.y * layer;
    uint64_t pixel_offset = sizeof(float3) * (pixel.x + pixel.y * params->size.x);
    float3 *clr = (float3 *) ((uint64_t)(params->layer_buffers) + layer_offset + pixel_offset);
    return *clr;
}

__device__ void device::SetLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer, float3 color) {
    uint64_t layer_offset = sizeof(float3) * params->size.x * params->size.y * layer;
    uint64_t pixel_offset = sizeof(float3) * (pixel.x + pixel.y * params->size.x);
    float3 *clr = (float3 *) ((uint64_t)(params->layer_buffers) + layer_offset + pixel_offset);
    *clr = color; 
}

// ===== kernel functions =====

__global__ void device::RayTrace(TraceParams *params) {
    // compute which ray this thread should be tracing
    uint32_t ray_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (ray_index >= params->num_rays) return;
    Ray *ray = &(params->rays[ray_index]);

    // find the nearest object
    Intersection obj = NearestObj(ray, params);

    // DEBUG DEBUG DEBUG
    if (obj.ptr == NULL) {
        // nothing was hit, set background color
        SetLayerBuffer(params, ray->pixel, ray->layer, make_float3(0.0f, 0.0f, 0.0f));
        
    } else {
        // set the object color
        switch (obj.type) {
            case SPHERE:
                SetLayerBuffer(params, ray->pixel, ray->layer, ((Sphere *) obj.ptr)->surface.color);
                break;
        }
    }
    // DEBUG DEBUG DEBUG

    // TODO: compute direct lighting (sample lights)

    // TODO: generate importance rays

    // TODO: generate ambient rays
}
