#include "device.h"

MetaObject *device::meta_chunk = NULL;
uint64_t *device::light_list = NULL;
void *device::obj_chunk = NULL;
float3 *device::layer_buffers = NULL;
Ray *device::ray_packet = NULL;
TraceParams *device::trace_params = NULL;

void device::AllocateLayerBuffers() {
    // allocate one huge chunk for all the layer buffers
    CUDA_SAFE_CALL(cudaMalloc<float3>(&layer_buffers, sizeof(float3) * host::render.size.x * host::render.size.y * host::render.max_bounces));
    
    // zero out each layer buffer by copying zeros into the device memory for
    // each layer
    float3 *zeroed = (float3 *)malloc(sizeof(float3) * host::render.size.x * host::render.size.y);
    memset(zeroed, 0, sizeof(float3) * host::render.size.x * host::render.size.y);
    for (uint64_t i = 0; i < host::render.max_bounces; i++) {
        uint64_t layer_offset = sizeof(float3) * host::render.size.x * host::render.size.y * i;
        float3 *layer = (float3 *) ((uint64_t)(layer_buffers) + layer_offset);
        CUDA_SAFE_CALL(cudaMemcpy(layer, zeroed, sizeof(float3) * host::render.size.x * host::render.size.y, cudaMemcpyHostToDevice));
    }
}

void device::CollapseLayerBuffers() {
    // TODO
}

float3 *device::GetLayerBuffer(uint64_t layer) {
    // compute source address
	uint64_t offset = sizeof(float3) * host::render.size.x * host::render.size.y * layer;
	void *src = (void *) ((uint64_t)layer_buffers + offset);

	// allocate space on the host for the destination buffer
	float3 *dest = (float3 *)malloc(sizeof(float3) * host::render.size.x * host::render.size.y);
    CUDA_SAFE_CALL(cudaMemcpy(dest, src, sizeof(float3) * host::render.size.x * host::render.size.y, cudaMemcpyDeviceToHost));
    
    return dest;
}

void device::DestroyLayerBuffers() {
	CUDA_SAFE_CALL(cudaFree(layer_buffers));
	layer_buffers = NULL;
}

void device::CopySceneToDevice() {
    // allocate space for the meta chunk on the device
    CUDA_SAFE_CALL(cudaMalloc<MetaObject>(&meta_chunk, sizeof(MetaObject) * host::num_objs));

    // copy the meta chunk to the device
    CUDA_SAFE_CALL(cudaMemcpy(meta_chunk, host::meta_chunk, sizeof(MetaObject) * host::num_objs, cudaMemcpyHostToDevice));
    
    // allocate space for the light list on the device
    CUDA_SAFE_CALL(cudaMalloc<uint64_t>(&light_list, sizeof(uint64_t) * host::num_lights));
    
    // copy the light list to the device
    CUDA_SAFE_CALL(cudaMemcpy(light_list, host::light_list, sizeof(uint64_t) * host::num_lights, cudaMemcpyHostToDevice));
    
    // allocate space on the device for the object chunk
    CUDA_SAFE_CALL(cudaMalloc(&obj_chunk, host::obj_chunk_size));

    // copy the object chunk to the device
    CUDA_SAFE_CALL(cudaMemcpy(obj_chunk, host::obj_chunk, host::obj_chunk_size, cudaMemcpyHostToDevice));
}

void device::RemoveSceneFromDevice() {
    // free device memory (if it was allocated)
    if (meta_chunk != NULL) {
    	CUDA_SAFE_CALL(cudaFree(meta_chunk));
    	meta_chunk = NULL;
   	}
   	if (light_list != NULL) {
   		CUDA_SAFE_CALL(cudaFree(light_list));
   		light_list = NULL;
   	}
    if (obj_chunk != NULL) {
    	CUDA_SAFE_CALL(cudaFree(obj_chunk));
    	obj_chunk = NULL;
   	}
}

void device::CopyRayPacketToDevice(Ray *packet, uint32_t num_rays) {
    // allocate space on the device
    CUDA_SAFE_CALL(cudaMalloc<Ray>(&ray_packet, sizeof(Ray) * num_rays));

    // copy the packet over
    CUDA_SAFE_CALL(cudaMemcpy(ray_packet, packet, sizeof(Ray) * num_rays, cudaMemcpyHostToDevice));
}

void device::RemoveRayPacketFromDevice() {
    // free device memory
    if (ray_packet != NULL) {
        CUDA_SAFE_CALL(cudaFree(ray_packet));
        ray_packet = NULL;
    }
}

void device::CopyTraceParamsToDevice(TraceParams *params) {
    // allocate memory on the device for it
    CUDA_SAFE_CALL(cudaMalloc<TraceParams>(&trace_params, sizeof(TraceParams)));

    // copy it to the device
    CUDA_SAFE_CALL(cudaMemcpy(trace_params, params, sizeof(TraceParams), cudaMemcpyHostToDevice));
}

void device::RemoveTraceParamsFromDevice() {
    // free device memory
    if (trace_params != NULL) {
        CUDA_SAFE_CALL(cudaFree(trace_params));
        trace_params = NULL;
    }
}
