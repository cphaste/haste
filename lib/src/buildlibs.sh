#!/bin/sh

# build lua
cd lib/src/LuaJIT-2.0.0-beta5
make
cp src/libluajit.a ../../linux/
cd ../../..
