#!/bin/bash

GCC_OPTS="-O3 -Wall -g"
mkdir -p "bin/static"
mkdir -p "bin/shared"
mkdir -p "bin/examples"

# object files
gcc $GCC_OPTS -c 	src/dnn.c -o bin/static/dnn.o -lm
gcc $GCC_OPTS -c -fPIC 	src/dnn.c -o bin/shared/dnn.o -lm

# library archives
ar rcs bin/static/libdnn.a bin/static/dnn.o
gcc -shared bin/shared/dnn.o -o bin/shared/libdnn.so

# to link to the static library:
# 	gcc main.c -Lbin/static -ldnn
# to link *dynamically* to the shared library
#	gcc main.c -Lbin/shared -ldnn
# LD_LIBRARY_PATH must be updated to include the shared
# library's path
#	LD_LIBRARY_PATH+=:($pwd)/bin/shared
# or the library can be installed to a defaut location:
#	cp bin/shared/libdnn.so /usr/local/lib64/

# compiling and linking examples

gcc $GCC_OPTS src/examples/main.c -Lbin/static -ldnn -o bin/examples/mnist -lm
