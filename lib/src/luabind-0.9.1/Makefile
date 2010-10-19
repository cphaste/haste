CXX = g++
AR = ar
TARGET = src/libluabind.a
INCLUDES = -I. -I../LuaJIT-2.0.0-beta5/src
CXXFLAGS = -O3 -ffast-math -Wall -msse2 $(INCLUDES)
LDFLAGS = -L../../linux
LIBS = -lm -ldl -lluajit

OBJS = src/class.o \
	   src/class_info.o \
	   src/class_registry.o \
	   src/class_rep.o \
	   src/create_class.o \
	   src/error.o \
	   src/exception_handler.o \
	   src/function.o \
	   src/inheritance.o \
	   src/link_compatibility.o \
	   src/object_rep.o \
	   src/open.o \
	   src/pcall.o \
	   src/scope.o \
	   src/stack_content_by_name.o \
	   src/weak_ref.o \
	   src/wrapper_base.o

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo Linking LuaBind
	$(AR) -rcs $(TARGET) $(OBJS)

clean:
	@echo Cleaning LuaBind
	rm -f $(TARGET)
	rm -f src/*.o

src/class.o: src/class.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/class_info.o: src/class_info.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/class_registry.o: src/class_registry.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/class_rep.o: src/class_rep.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/create_class.o: src/create_class.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/error.o: src/error.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/exception_handler.o: src/exception_handler.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/function.o: src/function.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/inheritance.o: src/inheritance.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/link_compatibility.o: src/link_compatibility.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/object_rep.o: src/object_rep.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/open.o: src/open.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/pcall.o: src/pcall.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/scope.o: src/scope.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/stack_content_by_name.o: src/stack_content_by_name.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/weak_ref.o: src/weak_ref.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

src/wrapper_base.o: src/wrapper_base.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

