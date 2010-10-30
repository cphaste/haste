#ifndef SCRIPTING_LUA_MACROS_H_
#define SCRIPTING_LUA_MACROS_H_

#include <stdint.h>
#include <lua/lua.hpp>

#include "global.h"
#include "scene.h"
#include "lua_extract.h"
#include "scene/objtypes.h"
#include "scene/sphere.h"

// render config extraction
int lua_macro_render(lua_State *L);

// geometry extraction
int lua_macro_sphere(lua_State *L);

#endif // SCRIPTING_LUA_MACROS_H_
