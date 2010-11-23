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

__device__ float3 device::operator*(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x * rhs.x;
    r.y = lhs.y * rhs.y;
    r.z = lhs.z * rhs.z;
    return r;
}

__device__ float3 device::operator/(const float3 &lhs, const float3 &rhs) {
    float3 r;
    r.x = lhs.x / rhs.x;
    r.y = lhs.y / rhs.y;
    r.z = lhs.z / rhs.z;
    return r;
}

__device__ float device::dot(const float3 &lhs, const float3 &rhs) {
    return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}

__device__ float device::length(const float3 &v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float3 device::normalize(const float3 &v) {
    float len = length(v);
    return v / make_float3(len, len, len);
}

__device__ float3 device::evaluate(Ray *ray, float t) {
    return ray->origin + (ray->direction * make_float3(t, t, t));
}

// ===== normal functions =====

__device__ float3 device::Normal(Sphere *sphere, const float3 &point) {
    float3 normal = point - sphere->position;
    return normalize(normal);
}

__device__ float3 device::Normal(Plane *plane, const float3 &point) {
    return normalize(plane->normal);
}

__device__ float3 device::Normal(Intersection *obj, const float3 &point) {
    // TODO: handle transforms here
    
    switch (obj->type) {
        case SPHERE:
            return Normal((Sphere *)(obj->ptr), point);
            
        case PLANE:
            return Normal((Plane *)(obj->ptr), point);
    }
    
    return make_float3(0.0f, 0.0f, 0.0f);
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

__device__ float device::Intersect(Ray *ray, Plane *plane) {
    // find p1 (a point on the plane) by using the formula for the point on the plane
    // closest to the origin
    float a = plane->normal.x;
    float b = plane->normal.y;
    float c = plane->normal.z;
    float d = plane->distance;
    float abc_sq = a * a + b * b + c * c;
    float3 p1 = make_float3((a * d) / abc_sq,
                             (b * d) / abc_sq,
                             (c * d) / abc_sq);
                             
    // solve for t
    float denom = dot(ray->direction, plane->normal);
    if (denom != 0.0f) {
        return dot((p1 - ray->origin), plane->normal) / denom;
    }

    // plane and ray are parallel, no intersection
    return -1.0f;
}

__device__ bool device::Intersect(Ray *ray, Intersection *obj) {
    // TODO: transform the ray by the inverse transformation matrix

    switch (obj->type) {
        case SPHERE:
            obj->t = Intersect(ray, (Sphere *)(obj->ptr));
            if (obj->t > EPSILON) return true;
            break;
            
        case PLANE:
            obj->t = Intersect(ray, (Plane *)(obj->ptr));
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
        if (obj.type == LIGHT) continue; // don't waste time on point lights
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
    // shift pixel coord into this slice's buffer space
    ushort2 pxl = make_ushort2(pixel.x - params->start, pixel.y);
    
    // calculate memory offsets
    uint64_t layer_offset = sizeof(float3) * params->width * params->render.size.y * layer;
    uint64_t pixel_offset = sizeof(float3) * (pxl.x + pxl.y * params->width);
    float3 *clr = (float3 *)((uint64_t)(params->layer_buffers) + layer_offset + pixel_offset);
    
    return *clr;
}

__device__ void device::SetLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer, float3 color) {
    // shift pixel coord into this slice's buffer space
    ushort2 pxl = make_ushort2(pixel.x - params->start, pixel.y);

    // calculate memory offsets
    uint64_t layer_offset = sizeof(float3) * params->width * params->render.size.y * layer;
    uint64_t pixel_offset = sizeof(float3) * (pxl.x + pxl.y * params->width);
    float3 *clr = (float3 *)((uint64_t)(params->layer_buffers) + layer_offset + pixel_offset);
    
    *clr = color; 
}

__device__ void device::BlendWithLayerBuffer(TraceParams *params, ushort2 pixel, uint64_t layer, float3 color) {
    // shift pixel coord into this slice's buffer space
    ushort2 pxl = make_ushort2(pixel.x - params->start, pixel.y);
    
    // calculate memory offsets
    uint64_t layer_offset = sizeof(float3) * params->width * params->render.size.y * layer;
    uint64_t pixel_offset = sizeof(float3) * (pxl.x + pxl.y * params->width);
    float *addr;
    
    // red component    
    addr = (float *)((uint64_t)(params->layer_buffers) + layer_offset + pixel_offset + (0 * sizeof(float)));
    atomicAdd(addr, color.x);
    
    // green component    
    addr = (float *)((uint64_t)(params->layer_buffers) + layer_offset + pixel_offset + (1 * sizeof(float)));
    atomicAdd(addr, color.y);
    
    // blue component    
    addr = (float *)((uint64_t)(params->layer_buffers) + layer_offset + pixel_offset + (2 * sizeof(float)));
    atomicAdd(addr, color.z);
}

__device__ Surface* device::GetSurface(Intersection *obj) {
    switch (obj->type) {
        case SPHERE:
            return &(((Sphere *)(obj->ptr))->surface);
            
        case PLANE:
            return &(((Plane *)(obj->ptr))->surface);
    }
    
    return NULL;
}

__device__ float3 device::GetLightColor(TraceParams *params, MetaObject *obj) {
    return ((Light *)((uint64_t)(params->obj_chunk) + obj->offset))->color;
}

__device__ float3 device::GetRandomLightPosition(TraceParams *params, MetaObject *obj) {
    switch (obj->type) {
        case LIGHT:
            Light *light = (Light *)((uint64_t)(params->obj_chunk) + obj->offset);
            return light->position;
            
        //case SPHERE:
            //return GetRandomLightPosition((Sphere *)((uint64_t)(params->obj_chunk) + obj->offset));
    }
    
    return make_float3(0.0f, 0.0f, 0.0f);
}

// ===== shading functions =====

__device__ void device::DirectShading(TraceParams *params, Ray *ray, Intersection *obj) {
    float3 hit_pt = evaluate(ray, obj->t);
    Surface *surface = GetSurface(obj);
    float3 N = Normal(obj, hit_pt);
    float contrib = ray->contrib * 1.0f / params->num_lights * 1.0f / params->render.direct_samples;
    float3 clr = {0.0f, 0.0f, 0.0f};
    

    // sample each light direct_samples times
    for (uint64_t i = 0; i < params->num_lights; i++) {
        uint64_t light_id = params->light_list[i];
        MetaObject light = params->meta_chunk[light_id];
        float3 light_clr = GetLightColor(params, &light);    
        
        // record each sample
        for (uint32_t j = 0; j < params->render.direct_samples; j++) {
            float3 light_pos = GetRandomLightPosition(params, &light);
            float3 L = normalize(light_pos - hit_pt);
            
            // diffuse component
            float NdotL = dot(N, L);
            NdotL = (NdotL > 0.0f) ? NdotL : 0.0f; // clamp to positive contributions only
            clr.x += contrib * surface->diffuse * surface->color.x * light_clr.x * NdotL;
            clr.y += contrib * surface->diffuse * surface->color.y * light_clr.y * NdotL;
            clr.z += contrib * surface->diffuse * surface->color.z * light_clr.z * NdotL;
            
            // specular component (half angle approximation)
            float3 H = normalize(L + (ray->direction * make_float3(-1.0f, -1.0f, -1.0f)));
            float NdotH = dot(N, H);
            NdotH = (NdotH > 0.0f) ? NdotH : 0.0f; // clamp to positive contributions only
            clr.x += contrib * surface->specular * light_clr.x * pow(NdotH, 1.0f / surface->shininess);
            clr.y += contrib * surface->specular * light_clr.y * pow(NdotH, 1.0f / surface->shininess);
            clr.z += contrib * surface->specular * light_clr.z * pow(NdotH, 1.0f / surface->shininess);
        }
    }
    
    // blend with the layer buffer
    BlendWithLayerBuffer(params, ray->pixel, ray->layer, clr);
}

// ===== kernel functions =====

__global__ void device::RayTrace(TraceParams *params) {
    // compute which ray this thread should be tracing
    uint32_t ray_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (ray_index >= params->num_rays) return;
    Ray *ray = &(params->rays[ray_index]);

    // find the nearest object
    Intersection obj = NearestObj(ray, params);

    // if the ray hit something...
    if (obj.ptr != NULL) {
        // compute direct lighting
        DirectShading(params, ray, &obj);

        // TODO: generate importance rays

        // TODO: generate ambient rays
    }
}
