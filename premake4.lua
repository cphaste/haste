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

-- generate nvcc prebuild commands for each .cu file, and file entries
-- for the generated .cu.cpp files
nvcccmds = {}
genfiles = {}
for i, cufile in ipairs(os.matchfiles("src/**.cu")) do
    local genfile = cufile .. ".cpp"
    table.insert(nvcccmds, "nvcc -cuda " .. cufile .. " -o " .. genfile)
    table.insert(genfiles, genfile)
end

-- global solution
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
            "src/**.cpp",
            genfiles
        }
        links {
            "luajit",
            "luabind",
            "cudart"
        }
        prebuildcommands {nvcccmds}
        uuid "56bdc40e-793c-4d8b-aeb0-ec1213fc391c"

-- custom action for building linux libraries
newaction {
    trigger = "buildlibs",
    description = "Build all libraries natively on Linux",
    execute = function() 
        if os.get() == "linux" then
            os.execute("./lib/src/buildlibs.sh")
        else
            print("Libraries are prebuilt for Win32. No build necessary.")
        end
    end
}

-- custom action for cleaning linux libraries
newaction {
    trigger = "cleanlibs",
    description = "Clean all native-built libraries on Linux",
    execute = function ()
        if os.get() == "linux" then
            os.execute("./lib/src/cleanlibs.sh")
        else
            print("Libraries are prebuilt for Win32. No need to clean.")
        end
    end
}

-- additional commands for "clean" action
if _ACTION == "clean" then
    -- remove any auto-generated CUDA .cu.cpp files
    print("Removing auto-generated CUDA source files...")
    for i, genfile in ipairs(os.matchfiles("src/**.cu.cpp")) do
        os.remove(genfile)
    end

    -- remove bin and obj directories and their contents
    print("Removing bin/ and obj/ directories...")
    os.rmdir("bin")
    os.rmdir("obj")
end
