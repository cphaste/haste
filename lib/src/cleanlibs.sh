#!/bin/sh

# delete generated libraries
rm lib/linux/*.a

# clean lua
cd lib/src/LuaJIT-2.0.0-beta5
make clean
cd ../../..
