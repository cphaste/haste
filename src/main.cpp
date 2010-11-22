#include "main.h"

int main(int argc, char *argv[]) {
    // check arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <scene.lua>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // load the scene file (lua script)
    printf("Loading scene file...\n");
    host::lstate = scripting::Init();
    scripting::Load(host::lstate, argv[1]);

    // get device properties and set up some options
    printf("Searching for CUDA devices...\n");
    std::vector<int> device_ids = host::QueryDevices();

    // create and launch control threads for each capable device
    printf("Launching %lu control threads...\n", device_ids.size());
    ControlThread **threads = (ControlThread **)malloc(sizeof(ControlThread *) * device_ids.size());
    uint16_t slice = host::render.size.x / device_ids.size();
    for (size_t i = 0; i < device_ids.size(); i++) {
        uint16_t start = i * slice;
        uint16_t width = (i == device_ids.size() - 1) ? host::render.size.x - start : slice;
    
        threads[i] = new ControlThread(device_ids[i], start, width);
        threads[i]->Start();
    }

    // wait for completion of device threads
    uint64_t ray_sum = 0;
    for (size_t i = 0; i < device_ids.size(); i++) {
        threads[i]->Join();
        ray_sum += threads[i]->total_rays();
    }
    printf("Control threads merged, %lu rays cast.\n", ray_sum);
    
    // merge into final image
    printf("Assembling final image...\n");
    float3 *img = (float3 *)malloc(sizeof(float3) * host::render.size.x * host::render.size.y);
    for (uint16_t x = 0; x < host::render.size.x; x++) {
        for (uint16_t y = 0; y < host::render.size.y; y++) {
            uint64_t dest_offset = (x + y * host::render.size.x) * sizeof(float3);
            float3 *dest = (float3 *)((uint64_t)img + dest_offset);
            
            int src_thread = x / slice;
            int src_thread_offset = x % slice;
            uint64_t src_offset = (src_thread_offset + y * threads[src_thread]->width()) * sizeof(float3);
            float3 *src = (float3 *)((uint64_t)threads[src_thread]->final() + src_offset);
            
            memcpy(dest, src, sizeof(float3));
        }
    }
    char *base_filename = host::GetOutputBaseName(argv[1]);
    image::Targa(img, host::render.size, base_filename);
    free(base_filename);
    free(img);

    // destroy control threads
    for (size_t i = 0; i < device_ids.size(); i++) {
        delete threads[i];
    }
    free(threads);
    
    // clean up the scene data
    printf("Cleaning up scene data on the host...\n");
    host::DestroyScene();

    // all done
    printf("Complete!\n");
    
    return EXIT_SUCCESS;
}
