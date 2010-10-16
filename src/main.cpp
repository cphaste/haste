#include "main.h"

int main(int argc, char *argv[]) {
    // check arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <main.lua>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    // create lua state
    //lua_State *lua = lua_open();
    //luabind::open(lua);
    lua_State *L = luaL_newstate();
    luaL_openlibs(L);
    
    // load the script
    luaL_dofile(L, argv[1]);

    // call saySomething()
    luabind::call_function<void>(L, "saySomething", "This string is in C++");
    
    printf("Goodbye from C++\n");

    return EXIT_SUCCESS;
}
