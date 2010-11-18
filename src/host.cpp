#include "host.h"

// initialize variables
Render host::render = DEFAULT_RENDER;
Camera host::camera = DEFAULT_CAMERA;
lua_State *host::lstate = NULL;
uint64_t host::num_objs = 0;
uint64_t host::num_lights = 0;
MetaObject *host::meta_chunk = NULL;
uint64_t *host::light_list = NULL;
void *host::obj_chunk = NULL;
uint64_t host::obj_chunk_size = 0;
std::queue<Ray *> host::ray_queue;
int host::num_blocks = 0;
int host::num_threads = 0;
uint32_t host::packet_size = 0;

inline static int ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    { { 0x10,  8 },
    { 0x11,  8 },
    { 0x12,  8 },
    { 0x13,  8 },
    { 0x20, 32 },
    { 0x21, 48 },
    {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    fprintf(stderr, "MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
    exit(EXIT_FAILURE);
    return -1;
}

void host::InsertIntoLightList(uint64_t id) {
	// expand the light list
	light_list = (uint64_t *)realloc(light_list, (num_lights + 1) * sizeof(uint64_t));
	
	// save the object id
	light_list[num_lights++] = id;
}

void host::DestroyScene() {
    // release host memory if it was allocated
    if (meta_chunk != NULL) {
        free(meta_chunk);
        meta_chunk = NULL;
    }
    if (obj_chunk != NULL) {
        free(obj_chunk);
        obj_chunk = NULL;
    }

    // reset counters
    obj_chunk_size = 0;
    num_objs = 0;
}

void host::QueryDevices() {
    int device_count = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        printf("FAILED.\n");
        fprintf(stderr, "No CUDA capable devices found!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp device_prop;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, i));

        int compute_cap_major = device_prop.major;
        int compute_cap_minor = device_prop.minor;
        int core_count = ConvertSMVer2Cores(compute_cap_major, compute_cap_minor) * device_prop.multiProcessorCount;
        float clock_speed = device_prop.clockRate * 1e-6f;
        float mem_size = device_prop.totalGlobalMem / 1024.0f / 1024.0f;

        // max out the number of threads per stream processor
        //num_threads = device_prop.maxThreadsPerBlock;
        // FOR SOME STRANGE REASON WE CAN'T GO OVER 320 THREADS PER BLOCK
        // IT RUNS, BUT GENERATES BLANK OUTPUT. WTF.
        num_threads = 320;

        // launch twice as many blocks as stream processors
        num_blocks = device_prop.multiProcessorCount * 2;

        // ideal packet size is one thread per ray
        packet_size = num_threads * num_blocks;

        printf("\n\tFound %s (%d.%d, %d cores, %.2f GHz, %.2f MB)",
            device_prop.name,
            compute_cap_major,
            compute_cap_minor,
            core_count,
            clock_speed,
            mem_size);
    }

    printf("\n");
}

void host::GeneratePrimaryRays() {
    // seed the random number generator
    srand(time(NULL));

    // compute bottom left of screen space extents
    float l = camera.aspect / -2.0f;
    float b = -0.5f;

    // compute camera's gaze vector
    float3 w = normalize(camera.look - camera.eye);

    // now compute the camera's up vector (not factoring in rotation yet)
    float3 temp = normalize(cross(w, camera.up));
    float3 v = normalize(cross(temp, w));

    // compute the point of the tip of the up vector
    float3 v_pt = camera.eye + v;
    
    // now rotate this point around the gaze vector
    // huge thanks to http://www.blitzbasic.com/Community/posts.php?topic=57616#645017 for this
    float3 rotated_v_pt;
    rotated_v_pt.x = w.x * (w.x * v_pt.x + w.y * v_pt.y + w.z * v_pt.z) + (v_pt.x * (w.y * w.y + w.z * w.z) - w.x * (w.y * v_pt.y + w.z * v_pt.z)) * cosf(camera.rotation * (float)M_PI / 180.0f) + (-w.z * v_pt.y + w.y * v_pt.z) * sinf(camera.rotation * (float)M_PI / 180.0f);
    rotated_v_pt.y = w.y * (w.x * v_pt.x + w.y * v_pt.y + w.z * v_pt.z) + (v_pt.y * (w.x * w.x + w.z * w.z) - w.y * (w.x * v_pt.x + w.z * v_pt.z)) * cosf(camera.rotation * (float)M_PI / 180.0f) + (w.z * v_pt.x - w.x * v_pt.z) * sinf(camera.rotation * (float)M_PI / 180.0f);
    rotated_v_pt.z = w.z * (w.x * v_pt.x + w.y * v_pt.y + w.z * v_pt.z) + (v_pt.z * (w.x * w.x + w.y * w.y) - w.z * (w.x * v_pt.x + w.y * v_pt.y)) * cosf(camera.rotation * (float)M_PI / 180.0f) + (-w.y * v_pt.x + w.x * v_pt.y) * sinf(camera.rotation * (float)M_PI / 180.0f);

    // recalculate the rotated up vector by subtracting off the eye position
    v = normalize(rotated_v_pt - camera.eye);

    // compute camera's u vector
    float3 u = normalize(cross(w, v));

    // loop over all pixels in the image (and each antialiasing cell per pixel)
    for (uint16_t x = 0; x < render.size.x; x++) {
        for (uint16_t y = 0; y < render.size.y; y++) {
            //printf("Pixel <%d, %d>:\n", x, y);
            for (uint32_t i = 0; i < render.antialiasing; i++) {
                for (uint32_t j = 0; j < render.antialiasing; j++) {
                    float us = 0.0f;
                    float vs = 0.0f;
                    float ws = 0.0f;
                    float contrib = 0.0f;
                    
                    // calculate screen space uvw
                    if (render.antialiasing <= 1) { // no antialiasing
                        us = l + (camera.aspect * (x + 0.5f) / render.size.x);
                        vs = b + (1.0f * (y + 0.5f) / render.size.y);
                        ws = 1.0f;      
                        contrib = 1.0f;              
                    } else {
                        //printf("\tRay <%d, %d>:\n", i, j);
                        float cell_size = 1.0f / render.antialiasing;
                        float rand_offset = (float)rand() / RAND_MAX;
                        us = l + (camera.aspect * (x + (i * cell_size) + (rand_offset * cell_size)) / render.size.x);
                        //printf("\t\tcell size = %f, rand_offset = %f\n", cell_size, rand_offset);
                        //printf("\t\ti * cell_size = %f, rand_offset * cell_size = %f\n", i * cell_size, rand_offset * cell_size);
                        rand_offset = (float)rand() / RAND_MAX;
                        vs = b + (1.0f * (y + (j * cell_size) + (rand_offset * cell_size)) / render.size.y);
                        //printf("\t\tj * cell_size = %f, rand_offset * cell_size = %f\n", j * cell_size, rand_offset * cell_size);
                        ws = 1.0f;
                        contrib = 1.0f / (render.antialiasing * render.antialiasing);
                        //printf("\t\tcontrib = %f\n", contrib);
                    }

                    // convert screen space point to world coords
                    float3 screen_pt = camera.eye +
                                       (u * make_float3(us, us, us)) +
                                       (v * make_float3(vs, vs, vs)) +
                                       (w * make_float3(ws, ws, ws));

                    // create ray
                    Ray *ray = (Ray *)malloc(sizeof(Ray));
                    ray->origin = camera.eye;
                    ray->direction = normalize(screen_pt - camera.eye);
                    ray->contrib = contrib;
                    ray->layer = 0;
                    ray->pixel = make_ushort2(x, y);
                    ray->unibounce = false;

                    // push it into the ray queue
                    ray_queue.push(ray);
                }
            }
        }
    }
}
