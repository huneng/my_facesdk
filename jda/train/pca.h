#ifndef _SDM_PCA_H_
#define _SDM_PCA_H_

#include "tool.h"

#include <lapacke.h>

#define PCA_DATA_AS_ROW 1
#define PCA_DATA_AS_COL 2

#define RET_OK         0
#define RET_SPEC       1
#define RET_IO_ERROR   2
#define RET_DATA_NULL  3
#define RET_FILE_ERROR 4

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

PCAModel* train_pca(float *feats, int cols, int rows, int step, float percent);

void release(PCAModel **model);

int save_pca_model(FILE *fout, PCAModel *model);
int load_pca_model(FILE *fin, PCAModel **ptrModel);
int load(const char *filePath, PCAModel **pcaModel);
int save(const char *filePath, PCAModel *model);

int projection(float *feats, int ssize, int dim, PCAModel* model, float *res);
void denoise(float *feats, int ssize, int dim, PCAModel *model, float factor = 3.0f);



#endif
