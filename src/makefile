CFLAGS=-O3 -Wall -ggdb --std=gnu99

libdanknn:	danknn.o
	cc -shared danknn.o -o libdanknn.so
	ar rcs libdanknn.a danknn.o

danknn.o:	danknn.c danknn.h danknn_intern.h
	cc $(CFLAGS) -c -fPIC danknn.c -o danknn.o -lm

.PHONY: clean
clean:
	-rm danknn.o libdanknn.so libdanknn.a
