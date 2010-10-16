#!/bin/sh

# build lua
cd lib/src/lua-5.1.4
make posix
cp src/liblua.a ../../linux/
cd ../../..
