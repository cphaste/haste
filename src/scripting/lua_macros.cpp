#include "lua_macros.h"

int lua_macro_render(lua_State *L) {
    // extract the render config from the first argument and set the host config
    lua_extract_render(L, 1, &host::render);

    // returns nothing
    return 0;
}

int lua_macro_sphere(lua_State *L) {
    // extract the sphere from the table
    Sphere sphere;
    lua_extract_sphere(L, 1, &sphere);

    // insert it into the host's scene
    uint64_t id = host::InsertIntoScene(SPHERE, &sphere);

    // push the table back onto the stack
    lua_pushvalue(L, 1);

    // set the id key
    lua_pushnumber(L, (lua_Number)id);
    lua_setfield(L, -2, "id");

    return 1;
}


