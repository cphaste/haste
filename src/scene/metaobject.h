#ifndef UTIL_METAOBJECT_H_
#define UTIL_METAOBJECT_H_

#include <stdint.h>

#include "objtypes.h"

struct MetaObject {
    uint32_t id;
    ObjType type;
    void *ptr;
};

typedef struct MetaObject MetaObject;

#endif // UTIL_METAOBJECT_H_
