#include "lua_macros.h"

int lua_macro_render(lua_State *L) {
    // extract the render config from the first argument and set the global
    lua_extract_render(L, 1, &global::render);

    // returns nothing
    return 0;
}

int lua_macro_sphere(lua_State *L) {
    // extract the sphere from the table
    Sphere sphere;
    lua_extract_sphere(L, 1, &sphere);

    // insert it into the global object table
    int32_t id = scene::Insert(SPHERE, &sphere);

    // push the table back onto the stack
    lua_pushvalue(L, 1);

    // set the id key
    lua_pushinteger(L, id);
    lua_setfield(L, -2, "id");

    return 1;
}


