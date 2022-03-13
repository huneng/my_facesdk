#ifndef _ALIGNER_H_
#define _ALIGNER_H_

#include "tool.h"

#define SAMPLE_IMAGE_SIZE 72
#define SAMPLE_IMAGE_SIZE2 145

typedef struct {
    float *vmean;
    float *vstd;

    float *eigenValues;
    float *eigenVectors;

    int dim;
    int featDim;

    int flag;
} PCAModel;



PCAModel* create_pca_data(int featDim, int dim);

void release(PCAModel **model);

int load_pca_model(FILE *fin, PCAModel **ptrModel);

void denoise(float *feats, int ssize, int dim, PCAModel *model, float factor = 3.0f);

void denoise_shape(PCAModel *model, Shape &shape, Shape &meanShape);


typedef struct {
    int stage;
    int isAlign;
    int ptsSize;
    int featDim;
    int oneDim;
    int gradDirect;

    Shape meanShape;

    int16_t **coeffs;

    PCAModel **rsdlPCA;
    PCAModel *shapePCA;
    PCAModel **pointsPCA;
} Aligner;


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, Shape &curShape);
void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, HRect &srect, Shape &curShape);

int load(const char *filePath, Aligner **raligner);

void release(Aligner **aligner);

#endif
