OBJ=cascade.o forest.o tree.o pca.o sample.o tool.o linear.o tron.o

DSTDIR=bin
FLAGS= -fopenmp -O3
#FLAGS= -fopenmp -g
DEPS = ${FLAGS} $(shell pkg-config --cflags --libs opencv lapacke) -lgfortran

TRAIN=${DSTDIR}/train
RATE=${DSTDIR}/rate
DETECT_VIDEO=${DSTDIR}/vdetect
DETECT_IMAGE=${DSTDIR}/idetect
GEN_SAMPLE=$(DSTDIR)/gen

DST=${TRAIN} ${DETECT_VIDEO} $(DETECT_IMAGE) $(RATE) $(GEN_SAMPLE)

all: outdir ${DST}

outdir:
	mkdir -p ${DSTDIR}

${TRAIN}: main.cpp ${OBJ}  
	g++ -DMAIN_TRAIN -o $@ $^ ${DEPS}

${RATE}: main.cpp ${OBJ}
	g++ -DMAIN_CALC_RATE -o $@ $^ ${DEPS}

${GEN_SAMPLE}: main.cpp ${OBJ}
	g++ -DMAIN_GENERATE_SAMPLES -o $@ $^ ${DEPS}

${DETECT_VIDEO}: main.cpp ${OBJ}
	g++ -DMAIN_DETECT_VIDEO -o $@ $^ ${DEPS}

${DETECT_IMAGE}: main.cpp ${OBJ}
	g++ -DMAIN_DETECT_IMAGES -o $@ $^ ${DEPS}

${OBJ}: %.o : %.cpp
	g++ -c ${FLAGS} $^


clean: 
	@rm -f ${OBJ} ${DST} 
