#ifndef _SAMPLE_H_
#define _SAMPLE_H_

#include "tool.h"

#define MAX_PTS_SIZE 101
#define MAX_PTS_SIZE2 (MAX_PTS_SIZE << 1)

#define FACTOR 3.0f

#define SAMPLE_IMAGE_SIZE 72
#define SAMPLE_FILE "image_samples.bin"


typedef struct{
    float scale;
    float angle;

    HPoint2f cen1, cen2;
} TranArgs;


typedef struct {
    float pts[MAX_PTS_SIZE2];
    int ptsSize;
} Shape;


typedef struct {
    Shape oriShape;
    uint8_t *img;
    char patchName[100];
} Sample;


typedef struct {
    Sample **samples;
    int ssize;
    int WINW;
    int WINH;

    Shape meanShape;
    int ptsSize;
} SampleSet;


int read_pts_file(const char *filePath, float *shape, int &ptsSize);
int write_pts_file(const char *filePath, float *shape, int ptsSize);

int read_mean_shape(const char *listFile, Shape &meanShape, int SHAPE_SIZE);
void similarity_transform(Shape &src, Shape &dst, TranArgs &arg);
int read_samples(const char *listFile, Shape &meanShape, int WINW, int WINH, int mirrorFlag, SampleSet **resSet);
int save(const char *filePath, SampleSet *set);
int load(const char *filePath, SampleSet **resSet);

void affine_shape(Shape &shapeSrc, Shape &shapeDst, TranArgs &arg);
void affine_sample(uint8_t *img, int width, int height, int stride, Shape &shape, TranArgs &arg);

HRect get_shape_rect(Shape &shape);

void release_data(SampleSet *set);
void release(SampleSet **set);

void show_rect(uint8_t *img, int width, int height, int stride, HRect &rect, const char *winName);
void show_shape(Shape &shape, const char *winName);
void show_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *winName);

void write_shape(Shape &shape, const char *filePath);
void write_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *filePath);

void write_images(SampleSet *set, int step, const char *outDir);
#endif
