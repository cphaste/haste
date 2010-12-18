#ifndef CONTROL_THREAD_H_
#define CONTROL_THREAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cutil.h>
#include <queue>
#include "ting/Thread.hpp"

#include "host.h"
#include "device/raytrace.cu.h"
#include "scene/metaobject.h"
#include "scene/lightobject.h"
#include "util/ray_packet.h"
#include "util/render.h"
#include "util/camera.h"
#include "util/vectors.h"
#include "util/ray.h"
#include "util/material.h"
#include "util/traceparams.h"

class ControlThread : public ting::Thread {
public:
    explicit ControlThread(int device_id, uint16_t start, uint16_t width);
    ~ControlThread();

    // thread entry point
    void Run();
    
    // the total number of rays cast by this thread, only valid after the thread
    // has run to completion
    inline uint64_t total_rays() const { return _total_rays; }

    // the starting x position of this thread
    inline uint16_t start() const { return _start; }
    
    // the width of the image slice that this thread rendered
    inline uint16_t width() const { return _width; }
    
    // pointer to raw image data for slice rendered by this thread, only
    // valid after it has run to completion
    inline float3 *final() const { return _final; }

private:
    int _device_id; // id of the CUDA device this thread is controlling
    uint16_t _start; // starting x position for this thread
    uint16_t _width; // width of the slice for this thread
    int _num_threads; // number of threads to launch each kernel with
    int _num_blocks; // number of blocks to launch each kernel with
    float3 *_layer_buffers; // base pointer to all the device's layer buffers
    curandState *_rand_states; // pointer to the device's states of randomness
    MetaObject *_meta_chunk; // base pointer to the device's meta chunk
    LightObject *_light_list; // base pointer to the device's list of light-emitting objects
    Material *_mat_list; // base pointer to the device's list of materials
    void *_obj_chunk; // base pointer to the device's object chunk
    std::queue<Ray *> _ray_queue; // queue of rays to be traced
    uint64_t _total_rays; // total count of all the rays traced by this thread
    Ray *_ray_chunk; // base pointer to the device's ray chunk
    float3 *_final; // image buffer for the final slice rendered by this thread
    
    // sets up the device for this thread
    void InitializeDevice();
    
    // allocate space on the device for everything
    void AllocateBuffers();
    
    // copies the host's data to the device
    void CopyDataToDevice();
    
    // clean up everything we allocated
    void CleanUp();
    
    // fills the ray queue with primary rays cast from the camera
    void GeneratePrimaryRays();
    
    // extract a layer buffer from the device (this allocates memory on the host,
    // which you should free() when you're done with it)
    float3 *GetLayerBuffer(uint64_t layer);
};

#endif // CONTROL_THREAD_H_
