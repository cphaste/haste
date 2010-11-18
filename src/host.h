#ifndef HOST_H_
#define HOST_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <queue>
#include <cuda_runtime.h>
#include <cutil.h>

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
    extern uint64_t num_objs; // number of objects in the scene
    extern uint64_t num_lights; // number of light emitting objects in the scene
    extern MetaObject *meta_chunk; // base pointer to the host's meta chunk
    extern uint64_t *light_list; // base pointer to the host's list of light-emitting objects
    extern void *obj_chunk; // base pointer to the host's object chunk
    extern uint64_t obj_chunk_size; // current size (in bytes) of the host's object chunk)
    extern std::queue<Ray *> ray_queue; // queue of rays to be traced
    extern int num_blocks; // the number of blocks to launch each kernel with
    extern int num_threads; // the number of threads to launch each kernel with
    extern uint32_t packet_size; // optimal number of rays in each ray packet

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
        uint64_t id = num_objs++;
        meta_chunk[id].id = id;
        meta_chunk[id].type = type;
        meta_chunk[id].offset = obj_chunk_size;

        // update the object chunk size
        obj_chunk_size += sizeof(T);

        return id;
    }
    
    // insert a new object into the lights
    void InsertIntoLightList(uint64_t id);

    // destroy the scene and free all memory
    void DestroyScene();

    // query devices and set up processing options
    void QueryDevices();

    // fills the ray queue with primary rays cast from the camera
    void GeneratePrimaryRays();
}

#endif // HOST_H_
