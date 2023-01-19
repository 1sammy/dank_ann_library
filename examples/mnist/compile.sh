#!/bin/bash

GCC_OPTS="-O3 -Wall -ggdb -flto"

gcc $GCC_OPTS main.c -L./ -ldanknn -o mnist -lm -pthread
