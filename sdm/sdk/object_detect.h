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
}ObjectDetector;


void init_detect_factor(ObjectDetector *cc, float startScale, float endScale, float offset, int layer);

int predict(ObjectDetector *cc, uint8_t *img, int width, int height, int stride);

int detect(ObjectDetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, float **rscores);

int load(ObjectDetector **cc, const char *filePath);
void release(ObjectDetector **cc);
#endif
