NVCC = nvcc

CFLAGS = -arch=sm_50 -Iinclude

SRC = main.cu kernel/warp_op.cu

OUTPUT = out.exe

all: $(OUTPUT)

$(OUTPUT): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(OUTPUT)

.PHONY: all clean
	