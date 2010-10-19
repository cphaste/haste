#include "add.cu.h"

// device code
namespace device {
    __global__ void Add(int a, int b, int *c) {
        *c = a + b;
    }
}

// host/device glue code
void gpu::Add(int a, int b, int *c) {
    device::Add<<<1, 1>>>(a, b, c);
}
