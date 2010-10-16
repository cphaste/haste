-- Premake script for generating build scripts
-- see http://industriousone.com/premake for more info

-- rebuild our libs if we're on linux
if _ACTION ~= "clean" and _ACTION ~= "cleanlibs" and os.get() == "linux" then
    os.execute("./lib/src/buildlibs.sh")
end

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
        libdirs { "lib/" .. os.get() }
        files {
            "src/**.cpp"
        }
        links {
            "m",
            "lua",
            "luabind"
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

-- additional actions
if _ACTION == "clean" then
    -- remove bin and obj directories and their contents
    print("Removing bin/ and obj/ directories...")
    os.rmdir("bin")
    os.rmdir("obj")
elseif _ACTION == "cleanlibs" and os.get() == "linux" then
    -- clean up the autobuilt libraries
    os.execute("./lib/src/cleanlibs.sh")
end
