#ifndef SCENE_LUA_SCENE_H_
#define SCENE_LUA_SCENE_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <string>

#include <lua/lua.hpp>
#include <luabind/luabind.hpp>

#include "global.h"
#include "util/uncopyable.h"

class LuaScene : private Uncopyable {
public:
    // ctors and dtors
    explicit LuaScene(const char *scene_file);
    ~LuaScene() {}

    // import the scene data from lua into c++
    void Import();

private:
    lua_State *_state;
};

#endif
