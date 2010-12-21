#include "lua_extract.h"

void lua_extract_render(lua_State *L, int index, Render *dest) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for Render value");
    
    // extract size
    lua_getfield(L, index, "size");
    if (lua_isnil(L, -1)) {
        dest->size = DEFAULT_RENDER.size;
    } else {
        lua_extract_ushort2(L, -1, &((*dest).size));
    }
    lua_pop(L, 1);

    // extract max_bounces
    lua_getfield(L, index, "max_bounces");
    if (lua_isnil(L, -1)) {
        dest->max_bounces = DEFAULT_RENDER.max_bounces; 
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as max_bounces element of Render value");
        dest->max_bounces = (uint32_t)lua_tonumber(L, -1);
        if (dest->max_bounces < 1) luaL_error(L, "max_bounces cannot be less than 1");
    }
    
    // extract antialiasing size
    lua_getfield(L, index, "antialiasing");
    if (lua_isnil(L, -1)) {
        dest->antialiasing = DEFAULT_RENDER.antialiasing; 
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as antialiasing element of Render value");
        dest->antialiasing = (uint32_t)lua_tonumber(L, -1);
        if (dest->antialiasing < 1) luaL_error(L, "antialiasing cannot be less than 1");
    }
    
    // extract direct_samples
    lua_getfield(L, index, "direct_samples");
    if (lua_isnil(L, -1)) {
        dest->direct_samples = DEFAULT_RENDER.direct_samples; 
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as direct_samples element of Render value");
        dest->direct_samples = (uint32_t)lua_tonumber(L, -1);
        if (dest->direct_samples < 1) luaL_error(L, "direct_samples cannot be less than 1");
    }
    
    // extract indirect_samples
    lua_getfield(L, index, "indirect_samples");
    if (lua_isnil(L, -1)) {
        dest->indirect_samples = DEFAULT_RENDER.indirect_samples; 
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as indirect_samples element of Render value");
        dest->indirect_samples = (uint32_t)lua_tonumber(L, -1);
        if (dest->indirect_samples < 1) luaL_error(L, "indirect_samples cannot be less than 1");
    }
    
    // extract gamma correction factor
    lua_getfield(L, index, "gamma_correction");
    if (lua_isnil(L, -1)) {
        dest->gamma_correction = DEFAULT_RENDER.gamma_correction; 
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as gamma_correction element of Render value");
        dest->gamma_correction = (float)lua_tonumber(L, -1);
    }
}

void lua_extract_float2(lua_State *L, int index, float2 *dest) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for float2 value");

    // extract x value
    lua_rawgeti(L, index, 1);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as first element of float2 value");
    dest->x = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
    
    // extract y value
    lua_rawgeti(L, index, 2);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as second element of float2 value");
    dest->y = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
}

void lua_extract_float3(lua_State *L, int index, float3 *dest) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for float3 value");

    // extract x value
    lua_rawgeti(L, index, 1);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as first element of float3 value");
    dest->x = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
    
    // extract y value
    lua_rawgeti(L, index, 2);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as second element of float3 value");
    dest->y = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
    
    // extract z value
    lua_rawgeti(L, index, 3);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as third element of float3 value");
    dest->z = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
}

void lua_extract_float4(lua_State *L, int index, float4 *dest) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for float4 value");

    // extract x value
    lua_rawgeti(L, index, 1);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as first element of float4 value");
    dest->x = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
    
    // extract y value
    lua_rawgeti(L, index, 2);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as second element of float4 value");
    dest->y = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
    
    // extract z value
    lua_rawgeti(L, index, 3);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as third element of float4 value");
    dest->z = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
    
    // extract w value
    lua_rawgeti(L, index, 4);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as fourth element of float4 value");
    dest->w = (float)lua_tonumber(L, -1);
    lua_pop(L, 1);
}

void lua_extract_ushort2(lua_State *L, int index, ushort2 *dest) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for ushort2 value");

    // extract x value
    lua_rawgeti(L, index, 1);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as first element of ushort2 value");
    dest->x = (uint16_t)lua_tonumber(L, -1);
    lua_pop(L, 1);

    // extract y value
    lua_rawgeti(L, index, 2);
    if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as second element of ushort2 value");
    dest->y = (uint16_t)lua_tonumber(L, -1);
    lua_pop(L, 1);
}

void lua_extract_material(lua_State *L, int index, Material *dest) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for Material value");
    
    // extract color
    lua_getfield(L, index, "color");
    if (lua_isnil(L, -1)) {
        dest->color = DEFAULT_MATERIAL.color;
    } else {
        lua_extract_float3(L, -1, &((*dest).color));
    }
    lua_pop(L, 1);
    
    // extract emissive
    lua_getfield(L, index, "emissive");
    if (lua_isnil(L, -1)) {
        dest->emissive = DEFAULT_MATERIAL.emissive;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as emissive element of Material value");
        dest->emissive = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract ambient
    lua_getfield(L, index, "ambient");
    if (lua_isnil(L, -1)) {
        dest->ambient = DEFAULT_MATERIAL.ambient;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as ambient element of Material value");
        dest->ambient = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract diffuse
    lua_getfield(L, index, "diffuse");
    if (lua_isnil(L, -1)) {
        dest->diffuse = DEFAULT_MATERIAL.diffuse;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as diffuse element of Material value");
        dest->diffuse = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract specular
    lua_getfield(L, index, "specular");
    if (lua_isnil(L, -1)) {
        dest->specular = DEFAULT_MATERIAL.specular;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as specular element of Material value");
        dest->specular = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract shininess
    lua_getfield(L, index, "shininess");
    if (lua_isnil(L, -1)) {
        dest->shininess = DEFAULT_MATERIAL.shininess;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as shininess element of Material value");
        dest->shininess = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract reflective
    lua_getfield(L, index, "reflective");
    if (lua_isnil(L, -1)) {
        dest->reflective = DEFAULT_MATERIAL.reflective;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as reflective element of Material value");
        dest->reflective = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract transmissive
    lua_getfield(L, index, "transmissive");
    if (lua_isnil(L, -1)) {
        dest->transmissive = DEFAULT_MATERIAL.transmissive;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as transmissive element of Material value");
        dest->transmissive = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract ior
    lua_getfield(L, index, "ior");
    if (lua_isnil(L, -1)) {
        dest->ior = DEFAULT_MATERIAL.ior;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as ior element of Material value");
        dest->ior = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
}

void lua_extract_camera(lua_State *L, int index, Camera *dest) {
	if (!lua_istable(L, index)) luaL_error(L, "expected table for Camera value");
	
	// extract eye position
	lua_getfield(L, index, "eye");
    if (lua_isnil(L, -1)) {
        dest->eye = DEFAULT_CAMERA.eye;
    } else {
        lua_extract_float3(L, -1, &((*dest).eye));
    }
    lua_pop(L, 1);
	
	// extract look at position
	lua_getfield(L, index, "look");
    if (lua_isnil(L, -1)) {
        dest->look = DEFAULT_CAMERA.look;
    } else {
        lua_extract_float3(L, -1, &((*dest).look));
    }
    lua_pop(L, 1);
	
	// extract world up vector
	lua_getfield(L, index, "up");
    if (lua_isnil(L, -1)) {
        dest->up = DEFAULT_CAMERA.up;
    } else {
        lua_extract_float3(L, -1, &((*dest).up));
    }
    lua_pop(L, 1);
	
	// extract gaze rotation
	lua_getfield(L, index, "rotation");
    if (lua_isnil(L, -1)) {
        dest->rotation = DEFAULT_CAMERA.rotation;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as rotation element of Camera value");
        dest->rotation = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
	
	// extract aspect ratio
	lua_getfield(L, index, "aspect");
    if (lua_isnil(L, -1)) {
        dest->aspect = DEFAULT_CAMERA.aspect;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as aspect element of Camera value");
        dest->aspect = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
}

void lua_extract_light(lua_State *L, int index, Light *dest) {
	if (!lua_istable(L, index)) luaL_error(L, "expected table for Light value");
    
    // extract position
    lua_getfield(L, index, "position");
    if (lua_isnil(L, -1)) {
        dest->position = DEFAULT_LIGHT.position;
    } else {
        lua_extract_float3(L, -1, &((*dest).position));
    }
    lua_pop(L, 1);
    
    // extract radius
    lua_getfield(L, index, "color");
    if (lua_isnil(L, -1)) {
        dest->color = DEFAULT_LIGHT.color;
    } else {
        lua_extract_float3(L, -1, &((*dest).color));
    }
    lua_pop(L, 1);
}

void lua_extract_sphere(lua_State *L, int index, Sphere *dest, Material *mat) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for Sphere value");
    
    // extract position
    lua_getfield(L, index, "position");
    if (lua_isnil(L, -1)) {
        dest->position = DEFAULT_SPHERE.position;
    } else {
        lua_extract_float3(L, -1, &((*dest).position));
    }
    lua_pop(L, 1);
    
    // extract radius
    lua_getfield(L, index, "radius");
    if (lua_isnil(L, -1)) {
        dest->radius = DEFAULT_SPHERE.radius;
    } else {
        if (!lua_isnumber(L, -1)) luaL_error(L, "expected number as radius element of Sphere value");
        dest->radius = (float)lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    
    // extract material
    lua_extract_material(L, index, mat);
}

void lua_extract_triangle(lua_State *L, int index, Triangle *dest, Material *mat) {
    if (!lua_istable(L, index)) luaL_error(L, "expected table for Triangle value");
    
    // extract vertex1
    lua_getfield(L, index, "vertex1");
    if (lua_isnil(L, -1)) {
        dest->vertex1 = DEFAULT_TRIANGLE.vertex1;
    } else {
        lua_extract_float3(L, -1, &((*dest).vertex1));
    }
    lua_pop(L, 1);
    
    // extract vertex2
    lua_getfield(L, index, "vertex2");
    if (lua_isnil(L, -1)) {
        dest->vertex2 = DEFAULT_TRIANGLE.vertex2;
    } else {
        lua_extract_float3(L, -1, &((*dest).vertex2));
    }
    lua_pop(L, 1);
    
    // extract vertex3
    lua_getfield(L, index, "vertex3");
    if (lua_isnil(L, -1)) {
        dest->vertex3 = DEFAULT_TRIANGLE.vertex3;
    } else {
        lua_extract_float3(L, -1, &((*dest).vertex3));
    }
    lua_pop(L, 1);
    
    // extract normal1
    lua_getfield(L, index, "normal1");
    if (lua_isnil(L, -1)) {
        dest->normal1 = DEFAULT_TRIANGLE.normal1;
    } else {
        lua_extract_float3(L, -1, &((*dest).normal1));
    }
    lua_pop(L, 1);
    dest->normal1 = normalize(dest->normal1);
    
    // extract normal2
    lua_getfield(L, index, "normal2");
    if (lua_isnil(L, -1)) {
        dest->normal2 = DEFAULT_TRIANGLE.normal2;
    } else {
        lua_extract_float3(L, -1, &((*dest).normal2));
    }
    lua_pop(L, 1);
    dest->normal2 = normalize(dest->normal2);
    
    // extract normal3
    lua_getfield(L, index, "normal3");
    if (lua_isnil(L, -1)) {
        dest->normal3 = DEFAULT_TRIANGLE.normal3;
    } else {
        lua_extract_float3(L, -1, &((*dest).normal3));
    }
    lua_pop(L, 1);
    dest->normal3 = normalize(dest->normal3);
    
    // extract material
    lua_extract_material(L, index, mat);
}
