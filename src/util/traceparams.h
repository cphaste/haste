#ifndef UTIL_TRACEPARAMS_H_
#define UTIL_TRACEPARAMS_H_

#include <stdint.h>
#include <cuda_runtime.h>

#include "util/ray.h"
#include "scene/metaobject.h"

typedef struct TraceParams {
    Ray *rays; // packet of rays to trace (should contain one ray per thread)
    uint32_t num_rays; // number of rays in the packet
    MetaObject *meta_chunk; // pointer to the start of the device meta chunk
    void *obj_chunk; // pointer to the start of the device object chunk
    uint64_t num_objs; // number of objects in the scene
    float3 *layer_buffers; // pointer to the start of the device layer buffers
    ushort2 size; // size of the final image in pixels
} TraceParams;

#endif // UTIL_TRACEPARAMS_H_
