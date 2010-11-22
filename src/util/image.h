#ifndef UTIL_IMAGE_H_
#define UTIL_IMAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

namespace image {
    void Targa(float3 *buffer, ushort2 size, const char *file);
}

#endif // UTIL_IMAGE_H_
