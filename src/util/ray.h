#ifndef UTIL_RAY_H_
#define UTIL_RAY_H_

#include <stdint.h>

#include "vectors.h"

typedef struct ray {
    float3 origin;
    float3 direction;
    float contrib;
    int32_t layer;
    int2 pixel;
    bool unibounce;
} ray_t;

#endif // UTIL_RAY_H_
