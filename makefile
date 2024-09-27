# Compilers and related
ROCM_PATH     ?= /opt/rocm
ROCM_GPU      ?= $(INSTALLED_GPU)
INSTALLED_GPU = $(shell $(ROCM_PATH)/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1})
CFLAG         = -Ofast -g --offload-arch=$(ROCM_GPU)
LDFLAG        =

# HIP
COMPILER_HIP  = $(ROCM_PATH)/bin/hipcc
CFLAGS_HIP    = -D_HIP
LDFLAGS_HIP   =

# DUNE
LDFLAGS_DUNE = -ldunecommon -ldunegeometry

# UMFPack
LDFLAGS_UMFPACK = -lumfpack

# Source code
OBJS= main.o \
	read-vectors.o
all:
	@echo "===================================="
	@echo "              Building              "
	@echo "===================================="
	mkdir -p build & mkdir -p build/bin
	rsync -ru main.cpp read-vectors.cpp read-vectors.hpp makefile build/bin
	$(MAKE) -C build/bin ilu0Solver CC=$(COMPILER_HIP) CFLAGS=$(CFLAGS_HIP) LDFLAGS=$(LDFLAGS_HIP)
	cp build/bin/ILU0Solver ./WellSolver

ilu0Solver: $(OBJS)
	    $(CC) $(CFLAG) $(CFLAGS) $(LDFLAG) $(LDFLAGS) $(LDFLAGS_DUNE) $(LDFLAGS_UMFPACK) -o ILU0Solver $(OBJS)

%.o : %.cpp
	$(CC) -I/usr/include/suitesparse $(CFLAG) $(CFLAGS) -c $< -o $@

clean:
	rm -rf ./WellSolver build/bin
