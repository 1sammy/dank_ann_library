all:
	+$(MAKE) -C src/
	cp src/libdanknn.so ./

.PHONY: clean
clean:
	+$(MAKE) clean -C src/
	rm libdanknn.so
