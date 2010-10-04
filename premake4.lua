-- Premake script for generating build scripts
-- see http://industriousone.com/premake for more info

solution "CUDArt"
    configurations { "Debug", "Release" }
    
    -- Global debug settings
    configuration "Debug"
        defines { "DEBUG" }
        flags { "Symbols", "FloatFast", "ExtraWarnings", "EnableSSE2", "NoExceptions", "NoPCH" }
        targetdir "bin/Debug"
        
    -- Global release settings
    configuration "Release"
        defines { "RELEASE", "NDEBUG" }
        flags { "OptimizeSpeed", "FloatFast", "ExtraWarnings", "EnableSSE2", "NoExceptions", "NoPCH" }
        targetdir "bin/Release"
        
    -- Global Windows settings
    configuration "windows"
        defines { "WIN32", "_WIN32", "_USE_MATH_DEFINES" }
        includedirs { "include/msinttypes" }
    
    -- Global Linux settings
    configuration "linux"
        defines { "LINUX" }
        
    -- CUDArt project
    project "CUDArt"
        kind "ConsoleApp"
        language "C++"
        targetname "cudart"
        includedirs { "include" }
        libdirs { "lib/" .. os.get() }
        files {
            "src/**.h",
            "src/**.cpp"
        }
        uuid "56bdc40e-793c-4d8b-aeb0-ec1213fc391c"
        
-- Additional clean commands
if _ACTION == "clean" then
    -- Remove bin and obj directories and their contents
    print("Removing bin/ and obj/ directories...")
    os.rmdir("bin")
    os.rmdir("obj")
end