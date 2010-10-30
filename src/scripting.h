#ifndef SCRIPTING_SCRIPTING_H_
#define SCRIPTING_SCRIPTING_H_

#include <stdio.h>
#include <stdlib.h>
#include <lua/lua.hpp>

#include "scripting/lua_macros.h"

namespace scripting {
    lua_State *Init();
    void Load(lua_State *L, const char *script);
}

#endif // SCRIPTING_SCRIPTING_H_
