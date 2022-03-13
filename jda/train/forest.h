#ifndef _JDA_FOREST_H_
#define _JDA_FOREST_H_

#include "pca.h"
#include "tree.h"


#define NEG_IMAGES_FILE "neg_images.bin"
#define POS_SET_FILE "pos_samples.bin"
#define NEG_PATCH_LIST_FILE "log/neg_patches_list.txt"

typedef struct {
    float recall;
    float prob;
    float radius;

    float npRate;

    int depth;
    int treeSize;

    int flag;
} TrainParams;

typedef struct {
    Tree **trees;

    float *threshes;

    int capacity;
    int treeSize;
    int depth;

    float **offsets;

    float *rsdlMean;
    float *rsdlCoeff;

    int rsdlDim;

    float bias[MAX_PTS_SIZE * 2];
} Forest;


typedef struct{
    FILE *fin;
    int isize;
    int id;

    Forest *forests;
    int fsize;

    int dx, dy;
    int tflag;

    int maxCount;

    Shape meanShape;

    char filePath[256];
} NegGenerator;


int generate_negative_images(const char *listFile, const char *outfile);

void train(Forest *forest, SampleSet *posSet, SampleSet *negSet, TrainParams *params, NegGenerator *generator);

int predict(Forest *forest, int size, Shape &meanShape, uint8_t *img, int stride, Shape &shape, float &score);
int predict(Forest *forest, Shape &meanShape, uint8_t *img, int stride, Shape &shape, float &score);
int predict_one(Forest *forest, Shape &meanShape, uint8_t *img, int stride, Shape &shape, float &score);

void save(FILE *fout, int ptsSize, Forest *forest);
void load(FILE *fin, int ptsSize, Forest *forest);

void release_data(Forest *forest);
void release(Forest **forest);

void release_data(NegGenerator *generator);
void release(NegGenerator **generator);
#endif
