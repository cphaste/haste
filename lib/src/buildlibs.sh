#!/bin/sh

# build lua
cd lib/src/LuaJIT-2.0.0-beta5
make
cp src/libluajit.a ../../linux/
cd ../../..

# build luabind
cd lib/src/luabind-0.9.1
make
cp src/libluabind.a ../../linux/
cd ../../../
