#ifndef SCENE_LIGHTOBJECT_H_
#define SCENE_LIGHTOBJECT_H_

#include <stdint.h>

#include "objtypes.h"

typedef struct LightObject {
    ObjType type;
	uint64_t offset;
} LightObject;

#endif // SCENE_LIGHTOBJECT_H_
