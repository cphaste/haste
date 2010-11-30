#ifndef UTIL_TRACEPARAMS_H_
#define UTIL_TRACEPARAMS_H_

#include <stdint.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "util/ray.h"
#include "util/render.h"
#include "util/material.h"
#include "scene/metaobject.h"
#include "scene/lightobject.h"

typedef struct TraceParams {
    Render render; // render configuration options
    uint16_t start; // start pixel (on the x axis) for multi-device rendering
    uint16_t width; // width of the slice (on the x axis) for multi-device rendering
    Ray *rays; // packet of rays to trace (should contain one ray per thread)
    uint32_t num_rays; // number of rays in the packet
    MetaObject *meta_chunk; // pointer to the start of the device meta chunk
    uint64_t num_objs; // number of objects in the scene
    LightObject *light_list; // pointer to the list of light-emitting objects
    uint64_t num_lights; // number of light-emitting objects in the scene
    Material *mat_list; // pointer to the list of materials
    uint64_t num_mats; // number of unique materials in the scene
    void *obj_chunk; // pointer to the start of the device object chunk
    float3 *layer_buffers; // pointer to the start of the device layer buffers
    curandState *rand_states; // states of the randomness
} TraceParams;

#endif // UTIL_TRACEPARAMS_H_
