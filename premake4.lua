-- premake script for generating build scripts
-- see http://industriousone.com/premake for more info

-- handy function for determining if we're on a 64-bit system (LINUX ONLY!)
function sixtyFourBits()
	if os.get() ~= "linux" then return false end
    local f = assert(io.popen("uname -m", "r"))
    local s = assert(f:read("*a"))
    f:close()
    if string.find(s, "64", 0, true) ~= nil then
        return true
    end
    return false
end

-- rebuild our libs if we're on linux
if _ACTION ~= "clean" and _ACTION ~= "cleanlibs" and os.get() == "linux" then
    os.execute("./lib/src/buildlibs.sh")
end

solution "cudart"
    configurations { "Debug", "Release" }
    
    -- global debug settings
    configuration "Debug"
        defines { "DEBUG" }
        flags { "Symbols", "ExtraWarnings", "EnableSSE2", "NoPCH" }
        targetdir "bin/Debug"
        
    -- global release settings
    configuration "Release"
        defines { "RELEASE", "NDEBUG" }
        flags { "OptimizeSpeed", "FloatFast", "ExtraWarnings", "EnableSSE2", "NoPCH" }
        targetdir "bin/Release"
        
    -- global windows settings
    configuration "windows"
        defines { "WIN32", "_WIN32", "_USE_MATH_DEFINES" }
        includedirs { "include/msinttypes", "c:/boost_1_44_0" }
    
    -- global linux settings
    configuration "linux"
        defines { "LINUX" }
        includedirs { "/usr/local/cuda/include" }
		links { "m", "dl" }
        if sixtyFourBits() then
            libdirs { "/usr/local/cuda/lib64" }
        else
            libdirs { "/usr/local/cuda/lib" }
        end
        
    -- haste project
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
            "luajit",
            "luabind",
            "cudart"
        }
        uuid "56bdc40e-793c-4d8b-aeb0-ec1213fc391c"

    -- luabind static library
    project "luabind"
        kind "StaticLib"
        language "C++"
        targetname "luabind"
        includedirs { "include" }
		libdirs { "lib/" .. os.get() }
        files {
            "luabind/**.cpp"
        }
        links {
            "luajit"
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
