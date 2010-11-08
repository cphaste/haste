#ifndef SCRIPTING_LUA_EXTRACT_H_
#define SCRIPTING_LUA_EXTRACT_H_

#include <stdint.h>
#include <cuda_runtime.h>
#include "lua/lua.hpp"

#include "defaults.h"
#include "util/render.h"
#include "util/surface.h"
#include "scene/sphere.h"

// render config extraction
void lua_extract_render(lua_State *L, int index, Render *dest);

// vector extractions
void lua_extract_float2(lua_State *L, int index, float2 *dest);
void lua_extract_float3(lua_State *L, int index, float3 *dest);
void lua_extract_float4(lua_State *L, int index, float4 *dest);
void lua_extract_ushort2(lua_State *L, int index, ushort2 *dest);

// surface extraction
void lua_extract_surface(lua_State *L, int index, Surface *dest);

// geometry extraction
void lua_extract_sphere(lua_State *L, int index, Sphere *dest);

#endif // SCRIPTING_LUA_EXTRACT_H_
