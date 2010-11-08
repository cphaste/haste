#ifndef UTIL_SURFACE_H_
#define UTIL_SURFACE_H_

#include <cuda_runtime.h>

typedef struct Surface {
    float3 color; // rgb color of the surface
    float emissive; // emissive lighting contribution (0.0 -> 1.0)
    float ambient; // ambient lighting contribution (0.0 -> 1.0)
    float diffuse; // diffuse lighting contribution (0.0 -> 1.0)
    float specular; // specular lighting contribution (0.0 -> 1.0)
    float shininess; // specular power (0.0 -> 1.0)
    float reflective; // reflective lighting contribution (0.0 -> 1.0)
    float transmissive; // transmissive (refractive) lighting contribution (0.0 -> 1.0)
    float ior; // index of refraction
} Surface;

#endif // UTIL_SURFACE_H_
