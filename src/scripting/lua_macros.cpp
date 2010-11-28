#include "lua_macros.h"

int lua_macro_render(lua_State *L) {
    // extract the render config from the first argument and set the host config
    lua_extract_render(L, 1, &host::render);

    // returns nothing
    return 0;
}

int lua_macro_camera(lua_State *L) {
	// extract the camera config from the first argument and set the host config
	lua_extract_camera(L, 1, &host::camera);
	
	// returns nothing
	return 0;
}

int lua_macro_light(lua_State *L) {
	// extract the light from the table
	Light light;
	lua_extract_light(L, 1, &light);
	
	// insert it into the host's scene
	uint64_t offset = host::InsertIntoScene(LIGHT, &light);
	
	// insert it into the light list
	host::InsertIntoLightList(LIGHT, offset);
	
	// push the table back onto the stack
	lua_pushvalue(L, 1);
	
	// set the id key
	lua_pushnumber(L, (lua_Number)offset);
	lua_setfield(L, -2, "id");
	
	return 1;
}

int lua_macro_sphere(lua_State *L) {
    // extract the sphere from the table
    Sphere sphere;
    Material mat;
    lua_extract_sphere(L, 1, &sphere, &mat);

    // insert it into the host's scene
    sphere.material = host::InsertIntoMaterialList(&mat);
    uint64_t offset = host::InsertIntoScene(SPHERE, &sphere);
    
    // if the sphere is emissive, insert it into the light list
    if (mat.emissive > 0.0f) {
    	host::InsertIntoLightList(SPHERE, offset);
    }

    // push the table back onto the stack
    lua_pushvalue(L, 1);

    // set the id key
    lua_pushnumber(L, (lua_Number)offset);
    lua_setfield(L, -2, "id");

    return 1;
}

int lua_macro_plane(lua_State *L) {
    // extract the plane from the table
    Plane plane;
    Material mat;
    lua_extract_plane(L, 1, &plane, &mat);

    // insert it into the host's scene
    plane.material = host::InsertIntoMaterialList(&mat);
    uint64_t offset = host::InsertIntoScene(PLANE, &plane);
    
    // if the plane is emissive, insert it into the light list
    if (mat.emissive > 0.0f) {
    	host::InsertIntoLightList(PLANE, offset);
    }

    // push the table back onto the stack
    lua_pushvalue(L, 1);

    // set the id key
    lua_pushnumber(L, (lua_Number)offset);
    lua_setfield(L, -2, "id");

    return 1;
}

int lua_macro_triangle(lua_State *L) {
    // extract the triangle from the table
    Triangle triangle;
    Material mat;
    lua_extract_triangle(L, 1, &triangle, &mat);

    // insert it into the host's scene
    triangle.material = host::InsertIntoMaterialList(&mat);
    uint64_t offset = host::InsertIntoScene(TRIANGLE, &triangle);
    
    // if the triangle is emissive, insert it into the light list
    if (mat.emissive > 0.0f) {
    	host::InsertIntoLightList(TRIANGLE, offset);
    }

    // push the table back onto the stack
    lua_pushvalue(L, 1);

    // set the id key
    lua_pushnumber(L, (lua_Number)offset);
    lua_setfield(L, -2, "id");

    return 1;
}
