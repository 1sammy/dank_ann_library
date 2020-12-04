#!/bin/bash

GCC_OPTS="-O3 -Wall -g"

gcc $GCC_OPTS main.c -L../../bin/static -ldnn -o mnist -lm
