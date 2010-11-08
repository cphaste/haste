#ifndef SCRIPTING_H_
#define SCRIPTING_H_

#include <stdio.h>
#include <stdlib.h>
#include "lua/lua.hpp"

#include "scripting/lua_extract.h"
#include "scripting/lua_macros.h"

namespace scripting {
    // initialize a new interpreter state for the scripting engine
    lua_State *Init();

    // load a script from a file and run it in the specified interpreter state
    void Load(lua_State *L, const char *script);
}

#endif // SCRIPTING_H_
