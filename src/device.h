#ifndef DEVICE_H_
#define DEVICE_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "host.h"
#include "scene/metaobject.h"
#include "util/ray.h"
#include "util/traceparams.h"

namespace device {
    extern MetaObject *meta_chunk; // base pointer to the device's meta chunk
    extern void *obj_chunk; // base pointer to the device's object chunk
    extern float3 *layer_buffers; // base pointer to all the device's layer buffers
    extern Ray *ray_packet; // base pointer to the device's ray packet
    extern TraceParams *trace_params; // base pointer to the device's trace parameters

    // allocate space on the device for the layer buffers
    void AllocateLayerBuffers();

    // collapse all the layer buffers into layer 0 (sum them)
    void CollapseLayerBuffers();

    // extract a layer buffer from the device (this allocates memory on the host,
    // which you should free() when you're done with it)
    float3 *GetLayerBuffer(uint64_t layer);

    // destroys all the layer buffers on the device
    void DestroyLayerBuffers();

    // copies the host's scene data to the device
    void CopySceneToDevice();

    // removes the scene data from the device
    void RemoveSceneFromDevice();

    // copies a ray packet to the device
    void CopyRayPacketToDevice(Ray *packet, uint32_t num_rays);

    // removes a ray packet from the device
    void RemoveRayPacketFromDevice();

    // copies trace parameters to the device
    void CopyTraceParamsToDevice(TraceParams *params);

    // removes trace parameters from the device
    void RemoveTraceParamsFromDevice();
}

#endif // DEVICE_H_
