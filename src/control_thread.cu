#include "control_thread.cu.h"

ControlThread::ControlThread(int device_id, uint16_t start, uint16_t width) :
    Thread(),
    _device_id(device_id),
    _start(start),
    _width(width),
    _num_threads(0),
    _num_blocks(0),
    _layer_buffers(NULL),
    _meta_chunk(NULL),
    _light_list(NULL),
    _obj_chunk(NULL),
    _total_rays(0),
    _ray_packet(NULL),
    _params(NULL),
    _final(NULL) {
    // empty
}

ControlThread::~ControlThread() {
    // release image buffer if it was allocated
    if (_final != NULL) {
        free(_final);
        _final = NULL;
    }
}

void ControlThread::Run() {
    // initialize the device for this thread
    printf("[%d] Initializing device...\n", _device_id);
    InitializeDevice();
    
    // allocate layer buffers on the gpu
    printf("[%d] Allocating layer buffers...\n", _device_id);
    AllocateLayerBuffers();

    // copy the scene data to the gpu
    printf("[%d] Copying scene data to device...\n", _device_id);
    CopySceneToDevice();

    // generate primary rays
    printf("[%d] Generating primary rays...\n", _device_id);
    GeneratePrimaryRays();

    // stream the rays down to the device
    printf("[%d] Starting trace...\n", _device_id);
    _total_rays = 0;
    uint32_t packet_size = _num_threads * _num_blocks;
    do {
        // extract next packet of rays
        uint32_t num_rays = (_ray_queue.size() >= packet_size) ? packet_size : _ray_queue.size();
        Ray *packet = (Ray *)malloc(sizeof(Ray) * num_rays);
        for (uint32_t i = 0; i < num_rays; i++) {
            memcpy(&(packet[i]), _ray_queue.front(), sizeof(Ray));
            free(_ray_queue.front());
            _ray_queue.pop();
        }

        // prepare the ray packet and trace parameters on the device
        PrepareForPacketTrace(packet, num_rays);

        // launch the kernel
        device::RayTrace<<<_num_blocks, _num_threads>>>(_params);

        // copy output rays back from the device
        // TODO

        // inject output rays into the ray queue
        // TODO

        // clean up
        RemovePacketTraceFromDevice();
        free(packet);
        
        // increment ray cast count
        _total_rays += num_rays;
    } while (_ray_queue.size() > 0);

    // collapse layer buffers
    // TODO
    
    // copy the base layer buffer into host memory as the final image
    _final = GetLayerBuffer(0);
    
    // remove the scene and layer buffers from the gpu
    printf("[%d] Cleaning up data on the device...\n", _device_id);
    RemoveSceneFromDevice();
    DestroyLayerBuffers();
    
    // control thread for this device finished
    printf("[%d] Finished.\n", _device_id);
}

void ControlThread::InitializeDevice() {
    // set the device
    CUDA_SAFE_CALL(cudaSetDevice(_device_id));
    
    // query the device to get the number of threads and blocks we should
    // launch kernels with
    cudaDeviceProp device_prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, _device_id));
    _num_threads = device_prop.maxThreadsPerBlock / 2;
    _num_blocks = device_prop.multiProcessorCount * 2;
}

void ControlThread::AllocateLayerBuffers() {
    // allocate one huge chunk for all the layer buffers
    CUDA_SAFE_CALL(cudaMalloc<float3>(&_layer_buffers, sizeof(float3) * _width * host::render.size.y * host::render.max_bounces));
    
    // zero out each layer buffer by copying zeros into the device memory for each layer
    float3 *zeroed = (float3 *)malloc(sizeof(float3) * _width * host::render.size.y);
    memset(zeroed, 0, sizeof(float3) * _width * host::render.size.y);
    for (uint64_t i = 0; i < host::render.max_bounces; i++) {
        uint64_t layer_offset = sizeof(float3) * _width * host::render.size.y * i;
        float3 *layer = (float3 *) ((uint64_t)(_layer_buffers) + layer_offset);
        CUDA_SAFE_CALL(cudaMemcpy(layer, zeroed, sizeof(float3) * _width * host::render.size.y, cudaMemcpyHostToDevice));
    }
    free(zeroed);
}

void ControlThread::DestroyLayerBuffers() {
    // free device memory (if it was allocated)
    if (_layer_buffers != NULL) {
	    CUDA_SAFE_CALL(cudaFree(_layer_buffers));
	    _layer_buffers = NULL;
    }
}

void ControlThread::CopySceneToDevice() {
    // allocate space for the meta chunk on the device
    CUDA_SAFE_CALL(cudaMalloc<MetaObject>(&_meta_chunk, sizeof(MetaObject) * host::num_objs));

    // copy the meta chunk to the device
    CUDA_SAFE_CALL(cudaMemcpy(_meta_chunk, host::meta_chunk, sizeof(MetaObject) * host::num_objs, cudaMemcpyHostToDevice));
    
    // allocate space for the light list on the device
    CUDA_SAFE_CALL(cudaMalloc<LightObject>(&_light_list, sizeof(LightObject) * host::num_lights));
    
    // copy the light list to the device
    CUDA_SAFE_CALL(cudaMemcpy(_light_list, host::light_list, sizeof(LightObject) * host::num_lights, cudaMemcpyHostToDevice));
    
    // allocate space on the device for the object chunk
    CUDA_SAFE_CALL(cudaMalloc(&_obj_chunk, host::obj_chunk_size));

    // copy the object chunk to the device
    CUDA_SAFE_CALL(cudaMemcpy(_obj_chunk, host::obj_chunk, host::obj_chunk_size, cudaMemcpyHostToDevice));
}

void ControlThread::RemoveSceneFromDevice() {
    // free device memory (if it was allocated)
    if (_meta_chunk != NULL) {
    	CUDA_SAFE_CALL(cudaFree(_meta_chunk));
    	_meta_chunk = NULL;
   	}
   	if (_light_list != NULL) {
   		CUDA_SAFE_CALL(cudaFree(_light_list));
   		_light_list = NULL;
   	}
    if (_obj_chunk != NULL) {
    	CUDA_SAFE_CALL(cudaFree(_obj_chunk));
    	_obj_chunk = NULL;
   	}
}

void ControlThread::GeneratePrimaryRays() {
    // seed the random number generator
    srand(time(NULL));

    // compute bottom left of screen space extents
    float l = host::camera.aspect / -2.0f;
    float b = -0.5f;

    // compute camera's gaze vector
    float3 w = normalize(host::camera.look - host::camera.eye);

    // now compute the camera's up vector (not factoring in rotation yet)
    float3 temp = normalize(cross(w, host::camera.up));
    float3 v = normalize(cross(temp, w));

    // compute the point of the tip of the up vector
    float3 v_pt = host::camera.eye + v;
    
    // now rotate this point around the gaze vector
    // huge thanks to http://www.blitzbasic.com/Community/posts.php?topic=57616#645017 for this
    float3 rotated_v_pt;
    rotated_v_pt.x = w.x * (w.x * v_pt.x + w.y * v_pt.y + w.z * v_pt.z) + (v_pt.x * (w.y * w.y + w.z * w.z) - w.x * (w.y * v_pt.y + w.z * v_pt.z)) * cosf(host::camera.rotation * (float)M_PI / 180.0f) + (-w.z * v_pt.y + w.y * v_pt.z) * sinf(host::camera.rotation * (float)M_PI / 180.0f);
    rotated_v_pt.y = w.y * (w.x * v_pt.x + w.y * v_pt.y + w.z * v_pt.z) + (v_pt.y * (w.x * w.x + w.z * w.z) - w.y * (w.x * v_pt.x + w.z * v_pt.z)) * cosf(host::camera.rotation * (float)M_PI / 180.0f) + (w.z * v_pt.x - w.x * v_pt.z) * sinf(host::camera.rotation * (float)M_PI / 180.0f);
    rotated_v_pt.z = w.z * (w.x * v_pt.x + w.y * v_pt.y + w.z * v_pt.z) + (v_pt.z * (w.x * w.x + w.y * w.y) - w.z * (w.x * v_pt.x + w.y * v_pt.y)) * cosf(host::camera.rotation * (float)M_PI / 180.0f) + (-w.y * v_pt.x + w.x * v_pt.y) * sinf(host::camera.rotation * (float)M_PI / 180.0f);

    // recalculate the rotated up vector by subtracting off the eye position
    v = normalize(rotated_v_pt - host::camera.eye);

    // compute camera's u vector
    float3 u = normalize(cross(w, v));

    // loop over all pixels in this slice (and each antialiasing cell per pixel)
    for (uint16_t x = _start; x < _start + _width; x++) {
        for (uint16_t y = 0; y < host::render.size.y; y++) {
            for (uint32_t i = 0; i < host::render.antialiasing; i++) {
                for (uint32_t j = 0; j < host::render.antialiasing; j++) {
                    float us = 0.0f;
                    float vs = 0.0f;
                    float ws = 0.0f;
                    float contrib = 0.0f;
                    
                    // calculate screen space uvw
                    if (host::render.antialiasing <= 1) { // no antialiasing
                        us = l + (host::camera.aspect * (x + 0.5f) / host::render.size.x);
                        vs = b + (1.0f * (y + 0.5f) / host::render.size.y);
                        ws = 1.0f;      
                        contrib = 1.0f;              
                    } else {
                        float cell_size = 1.0f / host::render.antialiasing;
                        float rand_offset = (float)rand() / RAND_MAX;
                        us = l + (host::camera.aspect * (x + (i * cell_size) + (rand_offset * cell_size)) / host::render.size.x);
                        rand_offset = (float)rand() / RAND_MAX;
                        vs = b + (1.0f * (y + (j * cell_size) + (rand_offset * cell_size)) / host::render.size.y);
                        ws = 1.0f;
                        contrib = 1.0f / (host::render.antialiasing * host::render.antialiasing);
                    }

                    // convert screen space point to world coords
                    float3 screen_pt = host::camera.eye +
                                       (u * make_float3(us, us, us)) +
                                       (v * make_float3(vs, vs, vs)) +
                                       (w * make_float3(ws, ws, ws));

                    // create ray
                    Ray *ray = (Ray *)malloc(sizeof(Ray));
                    ray->origin = host::camera.eye;
                    ray->direction = normalize(screen_pt - host::camera.eye);
                    ray->contrib = contrib;
                    ray->layer = 0;
                    ray->pixel = make_ushort2(x, y);
                    ray->unibounce = false;

                    // push it into the ray queue
                    _ray_queue.push(ray);
                }
            }
        }
    }
}

void ControlThread::PrepareForPacketTrace(Ray *packet, uint32_t num_rays) {
    // allocate space on the device
    CUDA_SAFE_CALL(cudaMalloc<Ray>(&_ray_packet, sizeof(Ray) * num_rays));

    // copy the packet over
    CUDA_SAFE_CALL(cudaMemcpy(_ray_packet, packet, sizeof(Ray) * num_rays, cudaMemcpyHostToDevice));
    
    // build trace parameters
    TraceParams params;
    params.rays = _ray_packet;
    params.num_rays = num_rays;
    params.render = host::render;
    params.meta_chunk = _meta_chunk;
    params.light_list = _light_list;
    params.obj_chunk = _obj_chunk;
    params.num_objs = host::num_objs;
    params.num_lights = host::num_lights;
    params.layer_buffers = _layer_buffers;
    params.start = _start;
    params.width = _width;
    
    // allocate memory on the device for it
    CUDA_SAFE_CALL(cudaMalloc<TraceParams>(&_params, sizeof(TraceParams)));

    // copy it to the device
    CUDA_SAFE_CALL(cudaMemcpy(_params, &params, sizeof(TraceParams), cudaMemcpyHostToDevice));
}

void ControlThread::RemovePacketTraceFromDevice() {
    // free device memory
    if (_ray_packet != NULL) {
        CUDA_SAFE_CALL(cudaFree(_ray_packet));
        _ray_packet = NULL;
    }
    if (_params != NULL) {
        CUDA_SAFE_CALL(cudaFree(_params));
        _params = NULL;
    }
}

float3 *ControlThread::GetLayerBuffer(uint64_t layer) {
    // compute source address
	uint64_t offset = sizeof(float3) * _width * host::render.size.y * layer;
	void *src = (void *)((uint64_t)_layer_buffers + offset);

	// allocate space on the host for the destination buffer
	float3 *dest = (float3 *)malloc(sizeof(float3) * _width * host::render.size.y);
    CUDA_SAFE_CALL(cudaMemcpy(dest, src, sizeof(float3) * _width * host::render.size.y, cudaMemcpyDeviceToHost));
    
    return dest;
}
