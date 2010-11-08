#!./lua

--[[

No need to edit this file unless you are changing something drastic
in the build process. You can just run it like so:

    ./build                 - build haste in default mode (debug at the moment)
    ./build debug           - build haste in debug mode
    ./build release         - build haste in release mode

If you have any questions, just ask me:
    rsomers@calpoly.edu

]]--

--------------------------------------------------------------------------
--                          MAKEFILE SETTINGS                           --
--------------------------------------------------------------------------

-- compiler
cxx = "nvcc"

-- binary names
bin = {}
bin.debug = "bin/Debug/haste"
bin.release = "bin/Debug/haste"

-- paths to search for including header files
includes = {
    "/home/bsomers/NVIDIA_GPU_Computing_SDK/C/common/inc",
    "include",
    "src"
}

-- preprocessor defines
defines = {}
defines.debug = {
    "DEBUG"
}
defines.release = {
    "RELEASE",
    "NDEBUG"
}

-- compiler options
cxxflags = {}
cxxflags.debug = {
    "c",                -- CUDA compile
    "m64",              -- 64-bit arch
    "g",                -- host debug symbols
    "G",                -- device debug symbols
    "pg",               -- gprof profiling
}
cxxflags.release = {
    "c",                -- CUDA compile
    "m64",              -- 64-bit arch
    "O2",               -- optimizer level 2
    "use_fast_math"     -- fast math library
}

-- object file directory
objdir = "obj"

-- static library directories
libdirs = {
    "lib"
}

-- static libraries to link
libs = {
    "dl",
    "luajit"
}

-- linker flags
ldflags = {
    "link",             -- CUDA link
    "m64"               -- 64-bit arch
}

--------------------------------------------------------------------------
--                   DO NOT EDIT BELOW THIS LINE!                       --
--------------------------------------------------------------------------

-- check command line arguments
mode = "debug"
if #arg > 1 then
    print "Usage: build [debug|release]"
    return
elseif #arg == 1 and (arg[1] == "-help" or arg[1] == "--help") then
    print "Usage: build [debug|release]"
    return
elseif #arg == 1 and arg[1] == "clean" then
    mode = "clean"
else
    if #arg == 1 then
        if arg[1] == "debug" or arg[1] == "release" then
            mode = arg[1]
        else
            print "Usage: build [debug|release]"
            return
        end
    end
end

-- bring in source files and process them
dofile "sources.lua"
for i, filename in ipairs(srcs) do
    local path, file = string.match(filename, "([^/]+)/(.+)")
    local extension = ""
    if path == nil then
        path = ""
        file, extension = string.match(filename, "([^.]+)\.(.+)")
    else
        file, extension = string.match(file, "([^.]+)\.(.+)")
    end
    if path == "" then
        srcs[i] = {path = "src" .. path, file = file, extension = extension}
    else
        srcs[i] = {path = "src/" .. path, file = file, extension = extension}
    end
end

-- CLEAN
if mode == "clean" then
    print("========== CLEANING EVERYTHING ==========")

    -- remove object files
    os.execute("rm " .. objdir .. "/*")

    -- remove binaries
    os.execute("rm " .. bin.debug .. " " .. bin.release)

    -- remove any generated makefiles
    os.execute("rm Makefile.debug Makefile.release")

    print("========== CLEAN COMPLETE ==========")
end

-- DEBUG
if mode == "debug" then
    print("========== DEBUG BUILD STARTING ==========")
    io.write("Generating Makefile.... ")
    io.flush()

    -- open output file
    local f = io.open("Makefile.debug", "w")

    -- compiler and binary name
    f:write("CXX = ", cxx, "\n")
    f:write("BIN = ", bin.debug, "\n")

    -- include directories
    includes.gen = ""
    if #includes > 0 then
        includes.gen = includes.gen .. " -I" .. table.concat(includes, " -I")
    end
    f:write("INCDIR =", includes.gen, "\n")

    -- preprocessor defines
    defines.gen = ""
    if #defines.debug > 0 then
        defines.gen = defines.gen .. " -D" .. table.concat(defines.debug, " -D")
    end
    f:write("DEFS =", defines.gen, "\n")

    -- library directory
    libdirs.gen = ""
    if #libdirs > 0 then
        libdirs.gen = " -L" .. table.concat(libdirs, " -L")
    end
    f:write("LIBDIRS =", libdirs.gen, "\n")

    -- static libraries
    libs.gen = ""
    if #libs > 0 then
        --libs.gen = libs.gen .. " $(LIBDIR)/" .. table.concat(libs, " $(LIBDIR)/")
        libs.gen = " -l" .. table.concat(libs, " -l")
    end
    f:write("LIBS =", libs.gen, "\n")

    -- compiler flags
    cxxflags.gen = ""
    if #cxxflags.debug > 0 then
        cxxflags.gen = cxxflags.gen .. " -" .. table.concat(cxxflags.debug, " -")
    end
    cxxflags.gen = cxxflags.gen .. " $(INCDIR) $(DEFS) -Xcompiler -Wall"
    f:write("CXXFLAGS =", cxxflags.gen, "\n")

    -- linker flags
    ldflags.gen = ""
    if #ldflags > 0 then
        ldflags.gen = ldflags.gen .. " -" .. table.concat(ldflags, " -")
    end
    f:write("LDFLAGS =", ldflags.gen, "\n")

    -- object files list
    local objs = ""
    for i, file in ipairs(srcs) do
        objs = table.concat({objs, " ", objdir, "/", file.file, ".o"})
    end
    f:write("OBJS =", objs, "\n")

    -- linker stage
    f:write("\nall: $(OBJS)\n")
    f:write("\t$(CXX) $(LDFLAGS) $(LIBDIRS) $(LIBS) $(OBJS) -o $(BIN)\n")

    -- compiler stage
    for i, file in ipairs(srcs) do
        local obj = table.concat({objdir, "/", file.file, ".o"})
        local src = table.concat({file.path, "/", file.file, ".", file.extension})
        f:write("\n", obj, ": ", src, "\n");
        f:write("\t$(CXX) $(CXXFLAGS) -o $@ ", src, "\n")
    end

    -- done
    f:close()
    print("done.")

    -- run makefile
    print("Running make...")
    os.execute("make -f Makefile.debug")

    print("========== DEBUG BUILD COMPLETE ==========")
end

-- RELEASE
if mode == "release" then
    print("========== RELEASE BUILD STARTING ==========")
    io.write("Generating Makefile.... ")
    io.flush()

    -- open output file
    local f = io.open("Makefile.release", "w")

    -- compiler and binary name
    f:write("CXX = ", cxx, "\n")
    f:write("BIN = ", bin.release, "\n")

    -- include directories
    includes.gen = ""
    if #includes > 0 then
        includes.gen = includes.gen .. " -I" .. table.concat(includes, " -I")
    end
    f:write("INCDIR =", includes.gen, "\n")

    -- preprocessor defines
    defines.gen = ""
    if #defines.release > 0 then
        defines.gen = defines.gen .. " -D" .. table.concat(defines.release, " -D")
    end
    f:write("DEFS =", defines.gen, "\n")

    -- library directory
    libdirs.gen = ""
    if #libdirs > 0 then
        libdirs.gen = " -L" .. table.concat(libdirs, " -L")
    end
    f:write("LIBDIRS =", libdirs.gen, "\n")

    -- static libraries
    libs.gen = ""
    if #libs > 0 then
        --libs.gen = libs.gen .. " $(LIBDIR)/" .. table.concat(libs, " $(LIBDIR)/")
        libs.gen = " -l" .. table.concat(libs, " -l")
    end
    f:write("LIBS =", libs.gen, "\n")

    -- compiler flags
    cxxflags.gen = ""
    if #cxxflags.release > 0 then
        cxxflags.gen = cxxflags.gen .. " -" .. table.concat(cxxflags.release, " -")
    end
    cxxflags.gen = cxxflags.gen .. " $(INCDIR) $(DEFS) -Xcompiler -Wall"
    f:write("CXXFLAGS =", cxxflags.gen, "\n")

    -- linker flags
    ldflags.gen = ""
    if #ldflags > 0 then
        ldflags.gen = ldflags.gen .. " -" .. table.concat(ldflags, " -")
    end
    f:write("LDFLAGS =", ldflags.gen, "\n")

    -- object files list
    local objs = ""
    for i, file in ipairs(srcs) do
        objs = table.concat({objs, " ", objdir, "/", file.file, ".o"})
    end
    f:write("OBJS =", objs, "\n")

    -- linker stage
    f:write("\nall: $(OBJS)\n")
    f:write("\t$(CXX) $(LDFLAGS) $(LIBDIRS) $(LIBS) $(OBJS) -o $(BIN)\n")

    -- compiler stage
    for i, file in ipairs(srcs) do
        local obj = table.concat({objdir, "/", file.file, ".o"})
        local src = table.concat({file.path, "/", file.file, ".", file.extension})
        f:write("\n", obj, ": ", src, "\n");
        f:write("\t$(CXX) $(CXXFLAGS) -o $@ ", src, "\n")
    end

    -- done
    f:close()
    print("done.")

    -- run makefile
    print("Running make...")
    os.execute("make -f Makefile.release")

    print("========== RELEASE BUILD COMPLETE ==========")
end

