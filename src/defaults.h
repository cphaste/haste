#ifndef DEFAULTS_H_
#define DEFAULTS_H_

#include "util/render.h"
#include "util/surface.h"
#include "scene/sphere.h"

extern "C" {
    const Render DEFAULT_RENDER = {
        {640, 480}             // size
    };

    const Surface DEFAULT_SURFACE = {
        {0.0f, 0.0f, 0.0f},     // color
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
        {0.0f, 0.0f, 0.3f},     // position
        1.0f,                   // radius
        DEFAULT_SURFACE,        // surface
    };
}

#endif // DEFAULTS_H_
