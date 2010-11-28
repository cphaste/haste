#ifndef HOST_H_
#define HOST_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <queue>
#include <cuda_runtime.h>
#include <cutil.h>
#include <sys/stat.h>
#include <dirent.h>

#include "defaults.h"
#include "scripting.h"
#include "util/render.h"
#include "util/camera.h"
#include "util/vectors.h"
#include "util/ray.h"
#include "scene/metaobject.h"

namespace host {
    extern Render render; // global render options
    extern Camera camera; // scene camera
    extern lua_State *lstate; // global lua interpreter state
    extern MetaObject *meta_chunk; // base pointer to the host's meta chunk
    extern uint64_t num_objs; // number of objects in the scene
    extern uint64_t *light_list; // base pointer to the host's list of light-emitting objects
    extern uint64_t num_lights; // number of light emitting objects in the scene
    extern void *obj_chunk; // base pointer to the host's object chunk
    extern uint64_t obj_chunk_size; // current size (in bytes) of the host's object chunk)

    // insert a new object into the scene
    template <typename T>
    uint64_t InsertIntoScene(ObjType type, T *obj) {
        // allocate space in the object chunk
        obj_chunk = realloc(obj_chunk, obj_chunk_size + sizeof(T));

        // copy the data into the object chunk
        T *dest = (T *) ((uint64_t)obj_chunk + obj_chunk_size);
        memcpy(dest, obj, sizeof(T));

        // allocate a new metaobject
        meta_chunk = (MetaObject *)realloc(meta_chunk, (num_objs + 1) * sizeof(MetaObject));

        // calculate id and populate meta object
        uint64_t id = obj_chunk_size;
        meta_chunk[num_objs].type = type;
        meta_chunk[num_objs].offset = obj_chunk_size;

        // update the number of objects and chunk size
        num_objs++;
        obj_chunk_size += sizeof(T);

        return id;
    }
    
    // insert a new object into the lights
    void InsertIntoLightList(uint64_t id);

    // destroy the scene and free all memory
    void DestroyScene();

    // returns a list of device id's for usable CUDA devices
    std::vector<int> QueryDevices();
    
    // returns the output file base name (no extension) by parsing the extension
    // off the input file name and returning a freshly allocated string (free
    // it when you're done)
    char *GetOutputBaseName(const char *input_filename);
}

#endif // HOST_H_
