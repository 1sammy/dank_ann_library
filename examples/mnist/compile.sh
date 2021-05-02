#!/bin/bash

GCC_OPTS="-O3 -Wall -g"

gcc $GCC_OPTS main.c -L./ -ldanknn -o mnist -lm
