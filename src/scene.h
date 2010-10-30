#ifndef SCENE_H_
#define SCENE_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "scripting.h"
#include "scene/metaobject.h"
#include "scene/objtypes.h"
#include "scene/sphere.h"

namespace scene {
    extern lua_State *lstate;
    extern int32_t num_objs;
    extern MetaObject *meta_chunk;
    extern void *obj_chunk;
    extern size_t obj_chunk_size;

    template <typename T>
    int32_t Insert(ObjType type, T *obj) {
        // make space at the end of the object chunk
        if (obj_chunk == NULL) {
            obj_chunk = malloc(sizeof(T));
        } else {
            obj_chunk = realloc(obj_chunk, obj_chunk_size + sizeof(T));
        }

        // copy the data into the object chunk
        T *dest = (T *) (((size_t) obj_chunk) + obj_chunk_size);
        memcpy(dest, obj, sizeof(T));

        // update the object chunk size
        obj_chunk_size += sizeof(T);

        // create new metaobject
        if (meta_chunk == NULL) {
            meta_chunk = (MetaObject *) malloc(sizeof(MetaObject));
        } else {
            meta_chunk = (MetaObject *) realloc(meta_chunk, (num_objs + 1) * sizeof(MetaObject));
        }

        // calculate id and populate meta object
        int32_t id = num_objs++;
        meta_chunk[id].id = id;
        meta_chunk[id].type = type;
        meta_chunk[id].ptr = (void *) dest;

        return id;
    }
}

#endif // SCENE_H_
