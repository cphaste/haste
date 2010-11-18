#include "main.cu.h"

int main(int argc, char *argv[]) {
    // check arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <scene.lua>\n", argv[0]);
        return EXIT_FAILURE;
    }
   
    // get device properties and set up some options
    printf("Querying CUDA devices... ");
    fflush(stdout);
    host::QueryDevices();
    printf("done.\n");

    // load the scene file (lua script)
    printf("Loading scene file... ");
    fflush(stdout);
    host::lstate = scripting::Init();
    scripting::Load(host::lstate, argv[1]);
    printf("done.\n");
    
    // allocate layer buffers on the gpu
    printf("Allocating layer buffers... ");
    fflush(stdout);
    device::AllocateLayerBuffers();
    printf("done.\n");

    // copy the scene data to the gpu
    printf("Copying scene data to device... ");
    fflush(stdout);
    device::CopySceneToDevice();
    printf("done.\n");

    // generate primary rays
    printf("Generating primary rays... ");
    fflush(stdout);
    host::GeneratePrimaryRays();
    printf("done.\n");

    // stream the rays down to the device
    printf("Beginning trace... ");
    fflush(stdout);
    do {
        // extract next packet of rays
        uint32_t num_rays = (host::ray_queue.size() >= host::packet_size) ? host::packet_size : host::ray_queue.size();
        Ray *packet = (Ray *)malloc(sizeof(Ray) * num_rays);
        for (uint32_t i = 0; i < num_rays; i++) {
            memcpy(&(packet[i]), host::ray_queue.front(), sizeof(Ray));
            free(host::ray_queue.front());
            host::ray_queue.pop();
        }

        // copy the rays to the device
        device::CopyRayPacketToDevice(packet, num_rays);

        // set up the trace parameters
        TraceParams params;
        params.rays = device::ray_packet;
        params.num_rays = num_rays;
        params.meta_chunk = device::meta_chunk;
        params.obj_chunk = device::obj_chunk;
        params.num_objs = host::num_objs;
        params.layer_buffers = device::layer_buffers;
        params.size = host::render.size;

        // copy the trace parameters to the device
        device::CopyTraceParamsToDevice(&params);

        printf("<threads, blocks, rays> = <%d, %d, %d>\n", host::num_threads, host::num_blocks, num_rays);

        // launch the kernel
        device::RayTrace<<<host::num_blocks, host::num_threads>>>(device::trace_params);

        // copy output rays back from the device
        // TODO

        // inject output rays into the ray queue
        // TODO

        // clean up
        device::RemoveTraceParamsFromDevice();
        device::RemoveRayPacketFromDevice();
        free(packet);
        
    } while (host::ray_queue.size() > 0);
    printf("done.\n");

    // collapse layer buffers
    // TODO

    // write image output
    printf("Writing image output... ");
    fflush(stdout);
    float3 *temp = device::GetLayerBuffer(0);
    image::Targa(temp, host::render.size, "initial_raycast.tga");
    free(temp);
    printf("done.\n");

    // remove the scene from the gpu
    printf("Cleaning up data on the device... ");
    fflush(stdout);
    device::RemoveSceneFromDevice();
    device::DestroyLayerBuffers();
    printf("done.\n");

    // clean up the scene data
    printf("Cleaning up data on the host... ");
    fflush(stdout);
    host::DestroyScene();
    printf("done.\n");

    // all done
    printf("Complete!\n");
    
    return EXIT_SUCCESS;
}
