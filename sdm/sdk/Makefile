OBJ = face_manager.o object_detect.o aligner.o tool.o

OUTDIR=bin

FLAGS = -O3 -fopenmp $(shell pkg-config --cflags opencv)
LIBS = $(shell pkg-config --libs opencv)

CXX = /usr/bin/g++-5

IDETECT=$(OUTDIR)/idetect
VDETECT=$(OUTDIR)/vdetect

DST = $(IDETECT) $(VDETECT)

all : outdir $(DST)

outdir: 
	mkdir -p $(OUTDIR)

$(IDETECT): main.cpp $(OBJ)
	$(CXX) -DMAIN_DETECT_IMAGE $^ -o $@ $(FLAGS) $(LIBS)

$(VDETECT): main.cpp $(OBJ)
	$(CXX) -DMAIN_DETECT_VIDEO $^ -o $@ $(FLAGS) $(LIBS)

$(OBJ):%.o:%.cpp
	$(CXX) -c $^ $(FLAGS)

clean:
	@rm -f $(OBJ)

