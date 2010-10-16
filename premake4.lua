-- Premake script for generating build scripts
-- see http://industriousone.com/premake for more info

solution "cudart"
    configurations { "Debug", "Release" }
    
    -- Global debug settings
    configuration "Debug"
        defines { "DEBUG" }
        flags { "Symbols", "FloatFast", "ExtraWarnings", "EnableSSE2", "NoPCH" }
        targetdir "bin/Debug"
        
    -- Global release settings
    configuration "Release"
        defines { "RELEASE", "NDEBUG" }
        flags { "OptimizeSpeed", "FloatFast", "ExtraWarnings", "EnableSSE2", "NoPCH" }
        targetdir "bin/Release"
        
    -- Global Windows settings
    configuration "windows"
        defines { "WIN32", "_WIN32", "_USE_MATH_DEFINES" }
        includedirs { "include/msinttypes" }
    
    -- Global Linux settings
    configuration "linux"
        defines { "LINUX" }
        
    -- Haste project
    project "haste"
        kind "ConsoleApp"
        language "C++"
        targetname "haste"
        includedirs { "include" }
        --libdirs { "lib/" .. os.get() }
        files {
            "src/**.cpp"
        }
        links {
            "luabind",
            "lua",
            "dl",
            "m"
        }
        uuid "56bdc40e-793c-4d8b-aeb0-ec1213fc391c"

    -- LuaBind static library
    project "luabind"
        kind "StaticLib"
        language "C++"
        targetname "luabind"
        includedirs { "include" }
        files {
            "luabind/**.cpp"
        }
        links {
            "lua"
        }
        uuid "cb2de148-1c2e-45ff-a558-f3eda7889614"

    -- Lua static library
    project "lua"
        kind "StaticLib"
        language "C"
        targetname "lua"
        defines { "LUA_USE_LINUX" }
        linkoptions { "-E" } 
        links {
            "m",
            "dl",
            "readline",
            "history",
            "ncurses"
        }
        includedirs { "lua" }
        files {
            "lua/lapi.c",
            "lua/lcode.c",
            "lua/ldebug.c",
            "lua/ldo.c",
            "lua/ldump.c",
            "lua/lfunc.c",
            "lua/lgc.c",
            "lua/llex.c",
            "lua/lmem.c",
            "lua/lobject.c",
            "lua/lopcodes.c",
            "lua/lparser.c",
            "lua/lstate.c",
            "lua/lstring.c",
            "lua/ltable.c",
            "lua/ltm.c",
            "lua/lundump.c",
            "lua/lvm.c",
            "lua/lzio.c",
            "lua/lauxlib.c",
            "lua/lbaselib.c",
            "lua/ldblib.c",
            "lua/liolib.c",
            "lua/lmathlib.c",
            "lua/loslib.c",
            "lua/ltablib.c",
            "lua/lstrlib.c",
            "lua/loadlib.c",
            "lua/linit.c"
        }
        uuid "d8587e09-e901-448e-9f1c-f7cb0e9e0de6"

-- Additional clean commands
if _ACTION == "clean" then
    -- Remove bin and obj directories and their contents
    print("Removing bin/ and obj/ directories...")
    os.rmdir("bin")
    os.rmdir("obj")
end
