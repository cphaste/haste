#include "main.h"

// Beginning of GPU Architecture definitions
/*inline int ConvertSMVer2Cores(int major, int minor)
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
    printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
    return -1;
}
// end of GPU Architecture definitions*/

int main(int argc, char *argv[]) {
    // check arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <main.lua>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    LuaScene lua_scene(argv[1]);
    lua_scene.Import();

    printf("Render config:\n");
    printf("\tWidth: %d\n", global::render.width);
    printf("\tHeight: %d\n", global::render.height);

    /*// create lua state
    lua_State *L = luaL_newstate();
    luaL_openlibs(L);
    
    // load the script
    luaL_dofile(L, argv[1]);

    // call saySomething()
    luabind::call_function<void>(L, "saySomething", "This string is in C++");
    
    luabind::object table = luabind::globals(L)["render_config"];
    if (luabind::type(table) == LUA_TTABLE) {
        for (luabind::iterator i(luabind::globals(L)["render_config"]), end; i != end; ++i) {
            printf("Key is: '%s'\n", luabind::object_cast<const char *>(i.key()));
        }
    } else {
        printf("Not a table!\n");
    }*/



    /*printf("Checking for CUDA devices...");

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed!\n");
        exit(EXIT_FAILURE);
    }

    printf(" found %d!\n", deviceCount);

    int totalCoreCount = 0;
    if (deviceCount > 0) {
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            
            printf("\n===== #%d: %s =====\n", i, deviceProp.name);

            int driverVersion, runtimeVersion;
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("\tDriver/Runtime Versions: %d.%d/%d.%d\n", driverVersion / 1000, driverVersion % 100, runtimeVersion / 1000, runtimeVersion % 100);

            printf("\tCompute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);

            int coreCount = ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
            printf("\tCore Count: %d\n", coreCount);
            totalCoreCount += coreCount;

            printf("\tClock Speed: %.2f GHz\n", deviceProp.clockRate * 1e-6f);

            float mem = deviceProp.totalGlobalMem / 1024.0 / 1024.0;
            printf("\tAvailable Memory: %.2f MB\n", mem);

            printf("\tMax Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
            printf("\tBlock Dimensions: %d x %d x %d threads\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
            printf("\tGrid Dimensions: %d x %d x %d blocks\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        }
    }

    printf("\nTotal CUDA Cores: %d\n", totalCoreCount);

    // run a cuda test on device 0
    if (deviceCount > 0) {
        printf("Selecting CUDA device #0... ");
        if (cudaSetDevice(0) != cudaSuccess) {
            fprintf(stderr, "ERROR: Problem with cudaSetDevice!\n");
            exit(EXIT_FAILURE);
        };
        printf("OK!\n");

        int c;
        int *dev_c;

        // allocate some memory on the device
        printf("Allocating memory on the device... ");
        if (cudaMalloc((void **)&dev_c, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "ERROR: Failure in cudaMalloc!\n");
            exit(EXIT_FAILURE);
        }
        printf("OK!\n");

        // run a test kernel on the device
        printf("Running a test kernel on the device... ");
        gpu::Add(2, 7, dev_c);
        if (cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "ERROR: Failure in cudaMemcpy!\n");
            exit(EXIT_FAILURE);
        }
        printf("2 + 7 = %d ", c);
        if (c == 9) {
            printf("OK!\n");
        } else {
            printf("fail\n");
        }

        // free some memory on the device
        printf("Freeing memory on the device... ");
        if (cudaFree(dev_c) != cudaSuccess) {
            fprintf(stderr, "ERROR: Failure in cudaFree!\n");
            exit(EXIT_FAILURE);
        }
        printf("OK!\n");
    }*/

    return EXIT_SUCCESS;
}
