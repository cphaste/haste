#!/bin/sh

# delete generated libraries
rm lib/linux/*.a

# clean lua
cd lib/src/lua-5.1.4
make clean
cd ../../..
