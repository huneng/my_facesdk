#ifndef _ALIGNER_FEATURE_H_
#define _ALIGNER_FEATURE_H_

#include "sample.h"
#include "pca.h"

#include "linear.h"

typedef struct FeatType_t
{
    uint8_t pntIdx1, pntIdx2;
    float off1X, off1Y;
    float off2X, off2Y;
} FeatType;


typedef struct JNode_t
{
    FeatType featType;
    int16_t thresh;

    struct JNode_t *left;
    struct JNode_t *right;

    uint8_t leafID;

    //for debug
    int posSize, negSize;
} JNode;


typedef JNode JTree;


typedef struct {
    JTree **trees;

    int treeSize;
    int depth;
    int ptsSize;


    float *offsets;
    float *vmean;
    float *coeff;

    int dim;

    //
    int featDim;
    int leafSize;
} Forest;

void train_forest(SampleSet *set, int treeSize, int depth, float radius, Forest *forest);
void predict(Forest *forest, uint8_t *img, int width, int height, int stride, Shape &curShape);

void release_data(Forest *forest);
void release(Forest **forest);

void load(FILE *fin, Forest **forest);
void save( FILE *fout, Forest *forest);

void mean_residual(float *rsdls, int ssize, int ptsSize);

#endif
