#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "defaults.h"
#include "scripting.h"
#include "util/render.h"
#include "scene/metaobject.h"
#include "scene/objtypes.h"
#include "scene/sphere.h"

namespace global {
    extern Render render;
}

#endif // GLOBAL_H_
