#ifndef _ALIGNER_H_
#define _ALIGNER_H_

#include "feature.h"


typedef struct {
    Forest **forests;

    Shape meanShape;

    int stage;
    int ptsSize;
} Aligner;


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, HRect &srect, Shape &curShape);
void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, Shape &curShape);

int load(const char *filePath, Aligner **aligner);

void release(Aligner **aligner);
#endif
