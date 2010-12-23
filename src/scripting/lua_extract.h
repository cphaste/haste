#ifndef SCRIPTING_LUA_EXTRACT_H_
#define SCRIPTING_LUA_EXTRACT_H_

#include <stdint.h>
#include <cuda_runtime.h>
#include "lua/lua.hpp"

#include "defaults.h"
#include "util/render.h"
#include "util/material.h"
#include "util/camera.h"
#include "util/vectors.h"
#include "scene/sphere.h"
#include "scene/triangle.h"

// render config extraction
void lua_extract_render(lua_State *L, int index, Render *dest);

// vector extractions
void lua_extract_float2(lua_State *L, int index, float2 *dest);
void lua_extract_float3(lua_State *L, int index, float3 *dest);
void lua_extract_float4(lua_State *L, int index, float4 *dest);
void lua_extract_ushort2(lua_State *L, int index, ushort2 *dest);

// material extraction
void lua_extract_material(lua_State *L, int index, Material *dest);

// camera extraction
void lua_extract_camera(lua_State *L, int index, Camera *dest);

// geometry extraction
void lua_extract_sphere(lua_State *L, int index, Sphere *dest, Material *mat);
void lua_extract_triangle(lua_State *L, int index, Triangle *dest, Material *mat);

#endif // SCRIPTING_LUA_EXTRACT_H_
