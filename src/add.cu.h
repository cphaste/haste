#ifndef GPU_ADD_H_
#define GPU_ADD_H_

#include <cuda_runtime.h>

// host/device glue code
namespace gpu {
    void Add(int a, int b, int *c);
}

#endif

