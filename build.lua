#!/usr/bin/lua

--[[

This is a custom build script to compile haste using nvcc, the CUDA compiler.
It uses some Lua logic to generate a Makefile and executes it automatically.
It can be run as follows:

    ./build.lua                 - build haste in default mode (debug at the moment)
    ./build.lua debug           - build haste in debug mode
    ./build.lua release         - build haste in release mode
    ./build.lua clean           - cleanup everything

If you have any questions, just ask me:
    bob@bobsomers.com

]]--

--------------------------------------------------------------------------
--                          MAKEFILE SETTINGS                           --
--------------------------------------------------------------------------

-- C++ and CUDA source files and directory
srcdir = "src"
srcs = {
    "control_thread.cu",
    "host.cpp",
    "main.cpp",
    "scripting.cpp",
    "device/raytrace.cu",
    "scripting/lua_extract.cpp",
    "scripting/lua_macros.cpp",
    "util/image.cpp",
}

-- binary names
bin = {
    debug = "hasteD",
    release = "haste"
}

-- paths to search for including header files
includes = {
    "/opt/nvidia/gpusdk/C/common/inc",
    "include",
    "src",

    debug = {
        -- nothing special
    },
    release = {
        -- nothing special
    }
}

-- preprocessor defines
defines = {
    debug = {
        "DEBUG"
    },
    release = {
        "RELEASE",
        "NDEBUG"
    }
}

-- compiler options
cflags = {
    "m64",                  -- 64-bit arch
    "arch=compute_20",      -- compute capability
    "code=sm_20",           -- device code generation version

    debug = {
        "g",                -- host debug symbols
        "G",                -- device debug symbols
        "pg",               -- gprof profiling
        "Xcompiler -Wall"   -- All warnings to g++, must be last
    },
    release = {
        "O3",               -- optimizer level 3
        "use_fast_math",    -- fast math library
        "Xcompiler -Wall"   -- All warnings to g++, must be last
    }
}

-- object files location
objdir = {
    debug = "obj/debug",
    release = "obj/release"
}

-- static library directories
libdirs = {
    "lib",
    
    debug = {
        -- nothing special
    },
    release = {
        -- nothing special
    }
}

-- static libraries to link
libs = {
    "m",                    -- math library
    "dl",                   -- dynamic linking
    "luajit",               -- fast lua
    
    debug = {
        -- nothing special
    },
    release = {
        -- nothing special
    }
}

-- linker flags
lflags = {
    "m64",                  -- 64-bit arch
    
    debug = {
        -- nothing special
    },
    release = {
        -- nothing special
    }
}

--------------------------------------------------------------------------
--                   DO NOT EDIT BELOW THIS LINE!                       --
--------------------------------------------------------------------------

-- function for generating nvcc command line options from tables
function genopts(both, additional, sep, desc)
    io.write(desc .. "... ")
    io.flush()
    
    local gen = ""
    
    if #both > 0 then
        gen = gen .. sep .. table.concat(both, sep)
    end
    
    if #additional > 0 then
        gen = gen .. sep .. table.concat(additional, sep)
    end
    
    print("done.")
    
    return gen
end

-- check command line arguments
mode = "debug"
if #arg > 1 then
    print "Usage: build [debug|release]"
    return
elseif #arg == 1 then
    if arg[1] == "-help" or arg[1] == "--help" then
        print "Usage: build [debug|release]"
    elseif arg[1] == "clean" or arg[1] == "debug" or arg[1] == "release" then
        mode = arg[1]
    else
        print "Usage: build [debug|release]"
    end
end

-- process source files
sources = {}
for i, v in ipairs(srcs) do
    local path, file = string.match(v, "([^/]+)/(.+)")
    local extension = ""
    if path == nil then
        path = ""
        file, extension = string.match(v, "([^.]+)\.(.+)")
    else
        file, extension = string.match(file, "([^.]+)\.(.+)")
    end
    if path == "" then
        sources[i] = {path = srcdir .. path, file = file, extension = extension}
    else
        sources[i] = {path = srcdir .. "/" .. path, file = file, extension = extension}
    end
end

-- CLEAN
if mode == "clean" then
    print("========== CLEANING EVERYTHING ==========")

    -- remove object files
    os.execute("rm " .. objdir.debug .. "/*.o")
    os.execute("rm " .. objdir.release .. "/*.o")

    -- remove binaries
    os.execute("rm " .. bin.debug .. " " .. bin.release)

    -- remove any generated makefiles
    os.execute("rm Makefile.debug Makefile.release")

    print("========== CLEAN COMPLETE ==========")
else
    if mode == "debug" then
        print("========== DEBUG BUILD STARTING ==========")
    
        -- build command line options for nvcc
        includes.gen = genopts(includes, includes.debug, " -I", "Include paths")
        defines.gen = genopts(defines, defines.debug, " -D", "Preprocessor defines")
        cflags.gen = " -c" .. genopts(cflags, cflags.debug, " -", "Compiler flags")
        libdirs.gen = genopts(libdirs, libdirs.debug, " -L", "Library paths")
        libs.gen = genopts(libs, libs.debug, " -l", "Static libraries")
        lflags.gen = " -link" .. genopts(lflags, lflags.debug, " -", "Linker flags")
    else
        print("========== RELEASE BUILD STARTING ==========")
    
        -- build command line options for nvcc
        includes.gen = genopts(includes, includes.release, " -I", "Include paths")
        defines.gen = genopts(defines, defines.release, " -D", "Preprocessor defines")
        cflags.gen = " -c" .. genopts(cflags, cflags.release, " -", "Compiler flags")
        libdirs.gen = genopts(libdirs, libdirs.release, " -L", "Library paths")
        libs.gen = genopts(libs, libs.release, " -l", "Static libraries")
        lflags.gen = " -link" .. genopts(lflags, lflags.release, " -", "Linker flags")
    end
    
    -- generate dependencies for each source file using nvcc
    print("Generating dependencies for:")
    for i, v in ipairs(sources) do
        local filename = table.concat({v.path, "/", v.file, ".", v.extension})
        
        io.write("\t" .. filename .. "... ")
        io.flush()
        
        local cmd = table.concat({"nvcc", includes.gen, " -M ", filename})
        local f = io.popen(cmd)
        sources[i].deps = f:read("*a")
        
        print("done.")
    end
    
    -- write out makefile
    io.write("Writing Makefile.... ")
    io.flush()

    -- open output file
    local makefilename = "Makefile." .. mode
    local f = io.open(makefilename, "w")

    -- build object files list
    local objpath = ""
    if mode == "debug" then
        objpath = objdir.debug
    else
        objpath = objdir.release
    end
    local objs = ""
    for i, v in ipairs(sources) do
        local filename = table.concat({objpath, "/", v.file, ".o"})
        objs = table.concat({objs, " ", filename})
    end

    -- write out "all" rule
    local binary = ""
    if mode == "debug" then
        binary = bin.debug
    else
        binary = bin.release
    end
    f:write(table.concat({"\nall:", objs, "\n"}))
    f:write(table.concat({"\tnvcc",
                          lflags.gen,
                          libdirs.gen,
                          libs.gen,
                          objs,
                          " -o ",
                          binary,
                          "\n\n"}))

    -- write out individual file rules
    for i, v in ipairs(sources) do
        local filename = table.concat({v.path, "/", v.file, ".", v.extension})
        f:write(table.concat({objpath, "/", v.deps}))
        f:write(table.concat({"\tnvcc",
                              cflags.gen,
                              includes.gen,
                              defines.gen,
                              " -o $@ ",
                              filename,
                              "\n\n"})) 
    end

    -- done
    f:close()
    print("done.")

    -- run makefile
    print("Running make...")
    local gofast = ""
    if mode == "release" then
        gofast = " -j 8"
    end
    os.execute("make -f " .. makefilename .. gofast)

    if mode == "debug" then
        print("========== DEBUG BUILD COMPLETE ==========")
    else
        print("========== RELEASE BUILD COMPLETE ==========")
    end
end

