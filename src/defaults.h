#ifndef DEFAULTS_H_
#define DEFAULTS_H_

#include "util/render.h"
#include "util/camera.h"
#include "util/material.h"
#include "scene/sphere.h"
#include "scene/triangle.h"

extern "C" {
    const Render DEFAULT_RENDER = {
        {640, 480},             // size
        1,                      // max_bounces
        1,                      // antialiasing
        10,                     // direct_samples
        10,                     // indirect_samples
        1.0f                    // gamma correction factor
    };

    const Camera DEFAULT_CAMERA = {
        {0.0f, 0.0f, 10.0f},    // eye position
        {0.0f, 0.0f, 0.0f},     // look at position
        {0.0f, 1.0f, 0.0f},     // up direction
        0.0f,                   // gaze vector rotation
        4.0f / 3.0f             // aspect ratio
    };

    const Material DEFAULT_MATERIAL = {
        {1.0f, 1.0f, 1.0f},     // color
        0.0f,                   // emissive
        0.2f,                   // ambient
        0.6f,                   // diffuse
        0.2f,                   // specular
        0.05f,                  // shininess
        0.0f,                   // reflective
        0.0f,                   // transmissive
        1.0f                    // ior
    };

    const Sphere DEFAULT_SPHERE = {
        {0.0f, 0.0f, 0.0f},     // position
        1.0f,                   // radius
        0                       // material
    };
    
    const Triangle DEFAULT_TRIANGLE = {
        {1.0f, -1.0f, 0.0f},    // vertex1
        {0.0f, 1.0f, 0.0f},     // vertex2
        {-1.0f, -1.0f, 0.0f},   // vertex3
        {0.0f, 0.0f, 1.0f},     // normal1
        {0.0f, 0.0f, 1.0f},     // normal2
        {0.0f, 0.0f, 1.0f},     // normal3
        0                       // material
    };
}

#endif // DEFAULTS_H_
