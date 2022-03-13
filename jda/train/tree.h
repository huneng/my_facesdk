#ifndef _JDA_TREE_H_
#define _JDA_TREE_H_

#include "sample.h"

#define CONF_NORM_FACTOR 2

typedef struct FeatType_t
{
    uint8_t pntIdx1, pntIdx2;
    float off1X, off1Y;
    float off2X, off2Y;
} FeatType;


typedef struct Node_t
{
    FeatType featType;
    int16_t thresh;

    struct Node_t *left;
    struct Node_t *right;

    uint8_t leafID;
    float score;

    //for debug
    int posSize, negSize;
    double pw, nw;
    uint8_t flag;
} Node;


typedef Node Tree;

float predict(Tree *root, uint8_t *img, int stride, Shape &shape, TranArgs &arg, uint8_t &leafID);
float predict(Tree *root, uint8_t *img, int stride, Shape &shape, uint8_t &leafID);

void save(FILE *fout, int depth, Tree *root);
void load(FILE *fin, int depth, Tree *root);

float train(Tree *root, int depth, SampleSet *posSet, SampleSet *negSet, float radius, float prob, int pntIdx, float recall);

void print_tree(FILE *fout, Tree *root, int depth);

void release(Tree **root);

#endif
