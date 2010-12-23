#ifndef SCRIPTING_LUA_MACROS_H_
#define SCRIPTING_LUA_MACROS_H_

#include <stdint.h>
#include "lua/lua.hpp"

#include "host.h"
#include "lua_extract.h"
#include "util/material.h"
#include "scene/objtypes.h"
#include "scene/sphere.h"
#include "scene/triangle.h"

// render configuration extraction
int lua_macro_render(lua_State *L);

// camera configuration extraction
int lua_macro_camera(lua_State *L);

// geometry extractions
int lua_macro_sphere(lua_State *L);
int lua_macro_triangle(lua_State *L);

#endif // SCRIPTING_LUA_MACROS_H_
