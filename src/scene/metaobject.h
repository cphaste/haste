#ifndef SCENE_METAOBJECT_H_
#define SCENE_METAOBJECT_H_

#include <stdint.h>

#include "objtypes.h"

typedef struct MetaObject {
    ObjType type;
	uint64_t offset;
} MetaObject;

#endif // SCENE_METAOBJECT_H_
