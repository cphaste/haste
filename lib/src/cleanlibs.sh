#!/bin/sh

# delete generated libraries
rm lib/linux/*.a

# clean lua
cd lib/src/LuaJIT-2.0.0-beta5
make clean
cd ../../..

# clean luabind
cd lib/src/luabind-0.9.1
make clean
cd ../../..
