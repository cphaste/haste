#include "scene.h"

lua_State *scene::lstate = NULL;
int32_t scene::num_objs = 0;
MetaObject *scene::meta_chunk = NULL;
void *scene::obj_chunk = NULL;
size_t scene::obj_chunk_size = 0;
