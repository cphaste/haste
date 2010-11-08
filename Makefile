# This Makefile is just a wrapper around the build script, but
# feel free to continue using make to build the project, it will work
# just fine.

# Source files do not go here! Add entries to the srcs table as necessary
# in "sources.lua". The correct Makefile syntax will be generated automatically!

all: debug

debug:
	./build.lua debug

release:
	./build.lua release

clean:
	./build.lua clean

help:
	@echo "Run \"make debug\" or \"make release\" to build debug or release versions of"
	@echo "haste. If you have any questions, ask Bob at rsomers@calpoly.edu!"
