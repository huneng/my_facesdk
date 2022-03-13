OBJ=joint_face.o

DSTDIR=bin
FLAGS= -fopenmp -O3
DEPS = ${FLAGS} -I/usr/local/include -I/usr/local/include/opencv -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect

RATE=${DSTDIR}/rate
DETECT_VIDEO=${DSTDIR}/vdetect
DETECT_IMAGE=${DSTDIR}/idetect

DST=${RATE} ${DETECT_VIDEO} $(DETECT_IMAGE)

all: outdir ${DST}

outdir:
	mkdir -p ${DSTDIR}

${RATE}: main.cpp ${OBJ}
	g++ -DMAIN_CALC_RATE -o $@ $^ ${DEPS}

${DETECT_VIDEO}: main.cpp ${OBJ}
	g++ -DMAIN_DETECT_VIDEO -o $@ $^ ${DEPS}

${DETECT_IMAGE}: main.cpp ${OBJ}
	g++ -DMAIN_DETECT_IMAGES -o $@ $^ ${DEPS}

${OBJ}: %.o : %.cpp
	g++ -c ${FLAGS} $^

clean: 
	@rm -f ${OBJ} ${DST} 
