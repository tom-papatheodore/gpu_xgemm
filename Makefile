#----------------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx90a

#----------------------------------------

INCLUDES  = -I${ROCM_PATH}/include
LIBRARIES = -L${ROCM_PATH}/lib -lhipblas -lrocblas

gpu_xgemm: gpu_xgemm.o
	${HIPCC} ${HIPFLAGS} ${LIBRARIES} gpu_xgemm.o -o gpu_xgemm

gpu_xgemm.o: gpu_xgemm.cpp
	${HIPCC} ${HIPFLAGS} ${INCLUDES} -c gpu_xgemm.cpp

.PHONY: clean

clean:
	rm -f gpu_xgemm *.o
