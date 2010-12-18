#ifndef UTIL_RAY_PACKET_H_
#define UTIL_RAY_PACKET_H_

#include <stdint.h>

#include "ray.h"

typedef struct RayPacket {
    Ray *rays; // packet of rays to trace (should contain one ray per thread)
    uint32_t num_rays; // number of rays in the packet
} RayPacket;

#endif // UTIL_RAY_PACKET_H_
