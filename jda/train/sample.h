#ifndef _JDA_SAMPLE_H_
#define _JDA_SAMPLE_H_

#include "tool.h"

#define PTS_SIZE 68
#define MAX_PTS_SIZE 68

#define SCALE_FACTOR 1.2f
#define BORDER_FACTOR 0.5f

typedef struct {
    float x;
    float y;
} HPoint2f;

#define TRANS_Q 14

typedef struct{
    float sina;
    float cosa;
    float scale;

    HPoint2f cen1, cen2;

    float ssina;
    float scosa;
} TranArgs;


typedef struct {
    HPoint2f pts[MAX_PTS_SIZE];
    int ptsSize;
} Shape;


typedef struct{
    uint8_t *iImg;
    uint8_t *img;
    int stride;


    Shape oriShape;
    Shape curShape;

    TranArgs arg;

    char patchName[100];

    float score;
} Sample;


void release_data(Sample *sample);
void release(Sample **sample);


typedef struct {
    Sample **samples;

    int ssize;
    int capacity;

    Shape meanShape;

    int ptsSize;

    int WINW;
    int WINH;
} SampleSet;

int write_pts_file(const char *filePath, Shape *shape);

void extract_sample_from_image(uint8_t *img, int width, int height, int stride, Shape &meanShape, int WINW, int WINH, int border, Sample* sample);

int read_mean_shape(const char *listFile, Shape &meanShape, int SHAPE_SIZE);
int read_positive_samples(const char *listFile, Shape meanShape, int WINW, int WINH, int mirrorFlag, SampleSet **resSet);
int read_negative_patch_samples(const char *listFile, SampleSet *negSet);
void generate_transform_samples(SampleSet *oriSet, int times, SampleSet *resSet);

void add_sample(SampleSet *set, Sample *sample);
void add_samples(SampleSet *dset, SampleSet *sset);
void add_sample_capacity_unchange(SampleSet *set, Sample *sample);

void similarity_transform(Shape &src, Shape &dst, TranArgs &arg);

int load(const char *filePath, SampleSet **resSet);
int save(const char *filePath, SampleSet *set);

void release_data(SampleSet *set);
void release(SampleSet **set);

void argument(Shape &shape, Shape &res);

void random_samples(SampleSet *set);

void calculate_residuals(SampleSet *set, double *rsdlsX, double *rsdlsY);
void calculate_residuals(SampleSet *set, int *idxs, int size, int pntIdx, float *rsdls);

void reset_scores_and_args(SampleSet *set);

void reserve(SampleSet *set, int capacity);

void refine_samples(SampleSet *set, float thresh, int flag);

void print_info(SampleSet *set);

void mirror_sample(uint8_t *img, int width, int height, int stride, Shape &shape);

void show_shape(Shape &shape, const char *winName);
void show_shape(cv::Mat &img, Shape &shape, const char *winName);
void show_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *winName);

void write_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *fileName);

void write_images(SampleSet *set, const char *outDir, int step);

void affine_shape(Shape *shapeSrc, HPoint2f cen1, Shape *shapeRes, HPoint2f cen2, float scale, float angle);
float shape_euler_distance(Shape &shapeA, Shape &shapeB);

void initial_positive_set_scores_by_distance(SampleSet *posSet);

#endif
