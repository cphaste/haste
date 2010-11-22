-- Listing of the source files needed to build haste
-- Note this is a listing of the cpp/cu files, not the object files

-- You do not need to specify the /src/ directory here. It's implied.
-- Just list the subdirectory, followed by the file's name. For example:
--
--      "device/raytrace.cu",
--
-- (note the trailing comma!)

-- By the way, double dashes are comments in Lua

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
