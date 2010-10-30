#ifndef UTIL_SURFACE_H_
#define UTIL_SURFACE_H_

#include "vectors.h"

struct Surface {
    float3 color;
    float emissive;
    float ambient;
    float diffuse;
    float specular;
    float shininess;
    float reflective;
    float transmissive;
    float ior;
};

typedef struct Surface Surface;

#endif // UTIL_SURFACE_H_
