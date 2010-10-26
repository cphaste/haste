#include "lua_scene.h"

LuaScene::LuaScene(const char *scene_file)
    : _state(NULL) {
    // create new lua interpreter state
    _state = luaL_newstate();
    luaL_openlibs(_state);

    // create expected globals for the lua script
    luabind::globals(_state)["render"] = luabind::newtable(_state);

    // load and run the scene file
    if (luaL_dofile(_state, scene_file)) {
        fprintf(stderr, "ERROR: Problem loading scene file.\n");
        fprintf(stderr, "%s\n", lua_tostring(_state, -1));
        exit(EXIT_FAILURE);
    }
}

void LuaScene::Import() {
    // import the render configuration data
    luabind::object render_table = luabind::globals(_state)["render"];
    if (luabind::type(render_table) != LUA_TTABLE) {
        fprintf(stderr, "ERROR: Global 'render' was redefined in the scene file.\n");
        exit(EXIT_FAILURE);
    }
    for (luabind::iterator i(render_table), end; i != end; ++i) {
        std::string key = luabind::object_cast<std::string>(i.key());
        if (key == "width") {
            if (luabind::type(*i) != LUA_TNUMBER) {
                fprintf(stderr, "ERROR: Global 'render.width' expected to be a number.\n");
                exit(EXIT_FAILURE);
            }
            global::render.width = luabind::object_cast<uint16_t>(*i);
        } else if (key == "height") {
            if (luabind::type(*i) != LUA_TNUMBER) {
                fprintf(stderr, "ERROR: Global 'render.height' expected to be a number.\n");
                exit(EXIT_FAILURE);
            }
            global::render.height = luabind::object_cast<uint16_t>(*i);
        }
    }
}
