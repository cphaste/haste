#ifndef UTIL_MATERIAL_H_
#define UTIL_MATERIAL_H_

#include <cuda_runtime.h>

typedef struct Material {
    float3 color; // rgb color of the material
    float emissive; // emissive lighting contribution (0.0 -> 1.0)
    float ambient; // ambient lighting contribution (0.0 -> 1.0)
    float diffuse; // diffuse lighting contribution (0.0 -> 1.0)
    float specular; // specular lighting contribution (0.0 -> 1.0)
    float shininess; // specular power (0.0 -> 1.0)
    float reflective; // reflective lighting contribution (0.0 -> 1.0)
    float transmissive; // transmissive (refractive) lighting contribution (0.0 -> 1.0)
    float ior; // index of refraction
} Material;

inline bool MaterialEqual(Material *mat1, Material *mat2) {
    if (mat1->color.x != mat2->color.x) goto not_equal;
    if (mat1->color.y != mat2->color.y) goto not_equal;
    if (mat1->color.z != mat2->color.z) goto not_equal;
    if (mat1->emissive != mat2->emissive) goto not_equal;
    if (mat1->ambient != mat2->ambient) goto not_equal;
    if (mat1->diffuse != mat2->diffuse) goto not_equal;
    if (mat1->specular != mat2->specular) goto not_equal;
    if (mat1->shininess != mat2->shininess) goto not_equal;
    if (mat1->reflective != mat2->reflective) goto not_equal;
    if (mat1->transmissive != mat2->transmissive) goto not_equal;
    if (mat1->ior != mat2->ior) goto not_equal;

    // everything matched
    return true;

not_equal:
    return false;
}

#endif // UTIL_MATERIAL_H_
