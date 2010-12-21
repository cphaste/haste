#include "scripting.h"

lua_State *scripting::Init() {
    // create new lua interpreter state
    lua_State *L = luaL_newstate();
    
    // open all the default lua libraries
    luaL_openlibs(L);

    // register macro callbacks
    lua_register(L, "render", lua_macro_render);
    lua_register(L, "camera", lua_macro_camera);
    lua_register(L, "light", lua_macro_light);
    lua_register(L, "sphere", lua_macro_sphere);
    lua_register(L, "triangle", lua_macro_triangle);

    return L;
}

void scripting::Load(lua_State *L, const char *script) {
    if (script != NULL) {
        if (luaL_dofile(L, script)) {
            fprintf(stderr, "%s\n", lua_tostring(L, -1));
            exit(EXIT_FAILURE);
        }
    }
}
