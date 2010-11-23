#ifndef DEFAULTS_H_
#define DEFAULTS_H_

#include "util/render.h"
#include "util/camera.h"
#include "util/surface.h"
#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/plane.h"

extern "C" {
    const Render DEFAULT_RENDER = {
        {640, 480},             // size
        1,                      // max_bounces
        1,                      // antialiasing
        10,                     // direct_samples
        10                      // indirect_samples
    };

    const Camera DEFAULT_CAMERA = {
        {0.0f, 0.0f, 10.0f},    // eye position
        {0.0f, 0.0f, 0.0f},     // look at position
        {0.0f, 1.0f, 0.0f},     // up direction
        0.0f,                   // gaze vector rotation
        4.0f / 3.0f             // aspect ratio
    };

    const Surface DEFAULT_SURFACE = {
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

	const Light DEFAULT_LIGHT = {
		{10.0f, 10.0f, 10.0f},	// position
		{1.0f, 1.0f, 1.0f}		// color
	};

    const Sphere DEFAULT_SPHERE = {
        {0.0f, 0.0f, 0.0f},     // position
        1.0f,                   // radius
        DEFAULT_SURFACE         // surface
    };
    
    const Plane DEFAULT_PLANE = {
        {0.0f, 1.0f, 0.0f},     // normal
        0.0f,                   // distance
        DEFAULT_SURFACE         // surface
    };
}

#endif // DEFAULTS_H_
