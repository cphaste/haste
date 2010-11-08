#ifndef UTIL_CAMERA_H_
#define UTIL_CAMERA_H_

#include <cuda_runtime.h>

typedef struct Camera {
    float3 eye; // eye position of the camera in world coords
    float3 look; // look at position of the camera in world coords
    float3 up; // world natural up direction (usually positive y)
    float rotation; // rotation (in degrees) about the gaze vector
    float aspect; // aspect ratio of the camera (width / height)
} Camera;

#endif // UTIL_CAMERA_H_