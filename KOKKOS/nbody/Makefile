KOKKOS_PATH = ../../../..
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
SRC = $(wildcard *.cpp)
vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3 -extended-lambda
LINK = ${CXX}
LDFLAGS = 
EXE = nbody.cuda
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "SNB,Maxwell52"
KOKKOS_ENABLE_CUDA=On
else
CXX = g++
CXXFLAGS = -O3
LINK = ${CXX}
LDFLAGS =  
EXE = nbody.host
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "SNB"
endif


DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
