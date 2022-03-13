OBJ = face_manager.o object_detect.o aligner.o pca.o feature.o sample.o tool.o linear.o tron.o

OUTDIR=bin

FLAGS = -O3 -fopenmp $(shell pkg-config --cflags opencv)
LIBS = $(shell pkg-config --libs opencv lapacke lapack openblas) /usr/lib/gcc/x86_64-linux-gnu/4.8/libgfortran.so

CXX = /usr/bin/g++-4.8

TRAIN=$(OUTDIR)/train
IDETECT=$(OUTDIR)/idetect
VDETECT=$(OUTDIR)/vdetect
GEN=$(OUTDIR)/gen

DST = $(TRAIN) $(IDETECT) $(VDETECT) $(GEN)

all : outdir $(DST)

outdir: 
	mkdir -p $(OUTDIR)

$(TRAIN): main.cpp $(OBJ)
	$(CXX) -DMAIN_TRAIN $^ -o $@ $(FLAGS) $(LIBS)

$(IDETECT): main.cpp $(OBJ)
	$(CXX) -DMAIN_DETECT_IMAGE $^ -o $@ $(FLAGS) $(LIBS)

$(VDETECT): main.cpp $(OBJ)
	$(CXX) -DMAIN_DETECT_VIDEO $^ -o $@ $(FLAGS) $(LIBS)

$(GEN): main.cpp $(OBJ)
	$(CXX) -DMAIN_GEN_SAMPLES $^ -o $@ $(FLAGS) $(LIBS)

$(OBJ):%.o:%.cpp
	$(CXX) -c $^ $(FLAGS)

clean:
	@rm -f $(OBJ)

