#ifndef _ALIGNER_H_
#define _ALIGNER_H_

#include "sample.h"
#include "pca.h"
#include "linear.h"
#include <omp.h>


typedef struct {
    int stage;
    int isAlign;
    int ptsSize;
    int featDim;
    int oneDim;
    int gradDirect;

    Shape meanShape;

    float **coeffs;

    PCAModel **rsdlPCA;
    PCAModel *shapePCA;

    PCAModel **pointsPCA;
} Aligner;


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, Shape &curShape);
void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, HRect &srect, Shape &curShape);

int load(const char *filePath, Aligner **raligner);
int save(const char *filePath, Aligner *aligner);

void release(Aligner **aligner);
void train(const char *listFile, int flag, int stage, Aligner **resAligner);

#endif
