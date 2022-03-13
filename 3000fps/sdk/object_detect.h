#ifndef _OBJECT_DETECT_H_
#define _OBJECT_DETECT_H_

#include "tool.h"

#define OBJECT_FACTOR 1.1f
typedef struct {
    uint8_t x00, y00;
    uint8_t x01, y01;
    uint8_t x10, y10;
    uint8_t x11, y11;
} FeatTemp;


typedef struct Node_t{
    float thresh;
    float score;

    FeatTemp ft;

    struct Node_t *lchild;
    struct Node_t *rchild;

} Node;


typedef Node Tree;


typedef struct {
    Tree **trees;

    int treeSize;

    int capacity;
    int depth;

    float *threshes;
} StrongClassifier;


typedef struct{
    StrongClassifier *sc;
    int WINW, WINH;
    int ssize;

    float startScale;
    float endScale;
    float offsetFactor;

    int layer;
}QTObjectDetector;


void init_detect_factor(QTObjectDetector *cc, float startScale, float endScale, float offset, int layer);

int predict(QTObjectDetector *cc, uint8_t *img, int width, int height, int stride);

int detect(QTObjectDetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, float **rscores);

int load(QTObjectDetector **cc, const char *filePath);
void release(QTObjectDetector **cc);
#endif
