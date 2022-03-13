#include "pca.h"


static void calculate_matrix_mean_values(float *matData, int width, int height, int flag, float *vmean){
    assert(matData != NULL && vmean != NULL);

    if(flag == PCA_DATA_AS_ROW){
        memset(vmean, 0, sizeof(float) * width);

        for(int y = 0; y < height; y++){
            float *ptrData = matData;
            float *ptrMean = vmean;

            int x;
            for(x = 0; x <= width - 4; x += 4){
                ptrMean[0] += ptrData[0];
                ptrMean[1] += ptrData[1];
                ptrMean[2] += ptrData[2];
                ptrMean[3] += ptrData[3];

                ptrMean += 4;
                ptrData += 4;
            }

            for(; x < width; x++){
                ptrMean[0] += ptrData[0];

                ptrMean ++;
                ptrData ++;
            }

            matData += width;
        }

        float factor = 1.0 / height;
        for(int x = 0; x < width; x++)
            vmean[x] *= factor;
    }
    else if(flag == PCA_DATA_AS_COL){

        memset(vmean, 0, sizeof(float) * height);

        for(int y = 0; y < height; y++){
            float t = 0;
            float *ptrData = matData;
            int x = 0;

            for(x = 0; x <= width - 4; x += 4){
                t += ptrData[0];
                t += ptrData[1];
                t += ptrData[2];
                t += ptrData[3];

                ptrData += 4;
            }

            for(; x < width; x++, ptrData++){
                t += ptrData[0];
            }

            vmean[y] = t / width;
            matData += width;
        }
    }
}


static void data_sub_mean(float *data, int width, int height, float *vmean, int flag){
    int x, y;

    assert(data != NULL && vmean != NULL);

    if(flag == PCA_DATA_AS_ROW){
        for(y = 0; y < height; y++){
            float *ptrMean = vmean;

            for(x = 0; x <= width - 4; x += 4){
                data[0] -= ptrMean[0];
                data[1] -= ptrMean[1];
                data[2] -= ptrMean[2];
                data[3] -= ptrMean[3];

                ptrMean += 4;
                data += 4;
            }

            for(; x < width; x++){
                data[0] -= ptrMean[0];
                ptrMean ++;
                data ++;
            }
        }
    }
    else if(flag == PCA_DATA_AS_COL){
        for(y = 0; y < height; y++){
            float t = vmean[y];
            for(x = 0; x <= width - 4; x += 4){
                data[0] -= t;
                data[1] -= t;
                data[2] -= t;
                data[3] -= t;

                data += 4;
            }

            for(; x < width; x ++){
                data[0] -= t;
                data ++;
            }
        }
    }
}



static void data_add_mean(float *data, int width, int height, float *vmean, int flag){
    int x, y;

    assert(data != NULL && vmean != NULL);

    if(flag == PCA_DATA_AS_ROW){
        for(y = 0; y < height; y++){
            float *ptrMean = vmean;

            for(x = 0; x <= width - 4; x += 4){
                data[0] += ptrMean[0];
                data[1] += ptrMean[1];
                data[2] += ptrMean[2];
                data[3] += ptrMean[3];

                ptrMean += 4;
                data += 4;
            }

            for(; x < width; x++){
                data[0] += ptrMean[0];
                ptrMean ++;
                data ++;
            }
        }
    }
    else if(flag == PCA_DATA_AS_COL){
        for(y = 0; y < height; y++){
            float t = vmean[y];
            for(x = 0; x <= width - 4; x += 4){
                data[0] += t;
                data[1] += t;
                data[2] += t;
                data[3] += t;

                data += 4;
            }

            for(; x < width; x ++){
                data[0] += t;
                data ++;
            }
        }
    }
}


void denoise(float *feats, float *eigenValues, int dim){
    for(int i = 0; i < dim; i++){
        feats[i] = HU_MIN(feats[i],  3 * eigenValues[i]);
        feats[i] = HU_MAX(feats[i], -3 * eigenValues[i]);
    }
}


void denoise(float *feats, int ssize, int dim, PCAModel *model, float factor){
    int featDim = model->featDim;

    float *arr = new float[dim];
    float *field = model->eigenValues;

    dim = HU_MIN(model->dim, dim);

    for(int i = 0; i < ssize; i++){
        float *coeff = model->eigenVectors;

        for(int d = 0; d < dim; d++){
            float sum = 0;

            for(int f = 0; f < featDim; f++)
                sum += feats[f] * coeff[f];

            float thresh = field[d] * factor;

            if(sum > thresh)
                sum = thresh;
            else if(sum < -thresh)
                sum = -thresh;
            arr[d] = sum;

            coeff += model->featDim;
        }


        memset(feats, 0, sizeof(float) * featDim);
        coeff = model->eigenVectors;

        for(int d = 0; d < dim; d++){
            float t = arr[d];

            for(int f = 0; f < featDim; f++)
                feats[f] += t * coeff[f];

            coeff += featDim;
        }

        feats += featDim;
    }

    feats -= featDim * ssize;


    delete [] arr;
}


void calculate_covariance(float *feats, int cols, int rows, float *cov){
    memset(cov, 0, sizeof(float) * cols * cols);

    for(int y = 0; y < rows; y++){
        float *ptrA = feats + y * cols;

        for(int x = 0; x < cols; x++){
            float *ptrB = cov + x * cols;
            float t = ptrA[x];

            for(int n = 0; n < cols; n++)
                ptrB[n] += t * ptrA[n];
        }
    }

    double num = 1.0 / (rows - 1);

    for(int y = 0; y < cols; y++)
        for(int x = 0; x < cols; x++)
            cov[y * cols + x] *= num;
}


static void eigen_mirror(float *veigen, float *eigenVectors, int dim){
    int len = dim >> 1;
    for(int i = 0; i < len; i++)
        HU_SWAP(veigen[i], veigen[dim - i - 1], float);

    //*
    float *buffer = new float[dim];

    for(int i = 0; i < len; i++){
        float *ptrA = eigenVectors + dim * i;
        float *ptrB = eigenVectors + dim * (dim - i - 1);

        memcpy(buffer, ptrA, sizeof(float) * dim);
        memcpy(ptrA, ptrB, sizeof(float) * dim);
        memcpy(ptrB, buffer, sizeof(float) * dim);
    }

    delete [] buffer;
    //*/
}


int symmetrix_eigen(float *covMat, int cols, int rows, int step, float *veigen, float *eigenVectors){
    int n = rows, lda = step, info, lwork, liwork;
    int iwkopt;
    int* iwork = NULL;
    float wkopt;
    float* work = NULL;

    char jobz[] = "Vectors";
    char uplo[] = "Upper";

    if (rows != cols || cols != step) {
        printf("DIMENTION NOT EQUAL\n");
        return 1;
    }

    if (covMat != eigenVectors)
        memcpy(eigenVectors, covMat, sizeof(float) * rows * cols);

    lwork = -1;
    liwork = -1;
    ssyevd_( jobz, uplo, &n, eigenVectors, &lda, veigen, &wkopt, &lwork, &iwkopt, &liwork, &info );

    lwork = (int)wkopt + 1;
    work = (float*)malloc( lwork*sizeof(float) ); assert(work != NULL);
    if (work == NULL) return RET_DATA_NULL;

    liwork = iwkopt + 1;
    iwork = (int*)malloc( liwork*sizeof(int) ); assert(iwork != NULL);
    if (iwork == NULL) {
        free(work);
        return RET_DATA_NULL;
    }

    ssyevd_( jobz, uplo, &n, eigenVectors, &lda, veigen, work, &lwork, iwork, &liwork, &info );

    if(info != 0){
        printf("info = %d\n", info);
        return RET_DATA_NULL;
    }

    free(iwork);
    free(work);

    return RET_OK;
}


PCAModel* create_pca_data(int featDim, int dim){
    PCAModel *model = new PCAModel;

    model->vmean = new float[featDim];
    model->vstd = new float[featDim];

    model->eigenValues = new float[dim];
    model->eigenVectors = new float[featDim * dim];

    model->dim = dim;
    model->featDim = featDim;

    model->flag = 0;

    return model;
}


PCAModel* train_pca(float *srcFeats, int cols, int rows, int step, float percent){
    int ret;
    PCAModel* model;

    assert(srcFeats != NULL);

    float *feats = new float[cols * rows];
    float *vmean = new float[cols];
    float *eigenValues = new float[cols];
    float *eigenVectors = new float[cols * cols];

    int dim;

    float num = 0, den = 0;

    for(int y = 0; y < rows; y++)
        memcpy(feats + y * cols, srcFeats + y * step, sizeof(float) * cols);

    calculate_matrix_mean_values(feats, cols, rows, PCA_DATA_AS_ROW, vmean);
    data_sub_mean(feats, cols, rows, vmean, PCA_DATA_AS_ROW);

    calculate_covariance(feats, cols, rows, eigenVectors);

    ret = symmetrix_eigen(eigenVectors, cols, cols, cols, eigenValues, eigenVectors); assert(ret == 0);

    eigen_mirror(eigenValues, eigenVectors, cols);

    for(int i = 0; i < cols; i++){
        if(eigenValues[i] <= 0) eigenValues[i] = 0;

        eigenValues[i] = sqrtf(eigenValues[i]);
        num += eigenValues[i];
    }

    for(int i = 0; i < cols; i++){
        float factor;
        den += eigenValues[i];
        factor = den / num;
        if(factor > percent){
            dim = i + 1;
            break;
        }
    }

    if(percent == 1) dim = cols;

    printf("percent = %f, featDim = %d, dim = %d\n", percent, cols, dim);
    model = create_pca_data(cols, dim);

    memcpy(model->vmean, vmean, sizeof(float) * cols);
    memcpy(model->eigenValues, eigenValues, sizeof(float) * dim);
    memcpy(model->eigenVectors, eigenVectors , sizeof(float) * dim * cols);

    model->flag = 0;

    delete [] feats;
    delete [] vmean;
    delete [] eigenValues;
    delete [] eigenVectors;

    return model;
}


int projection(float *feats, int ssize, int dim, PCAModel* model, float *res){
    int featDim = model->featDim;
    float *buffer = new float[featDim];

    dim = HU_MIN(dim, model->dim);

    for(int i = 0; i < ssize; i++){
        float *coeff = model->eigenVectors;

        memcpy(buffer, feats, sizeof(float) * featDim);

        for(int j = 0; j < featDim; j++)
            buffer[j] -= model->vmean[j];

        memset(res, 0, sizeof(float) * dim);

        for(int y = 0; y < dim; y++){
            for(int x = 0; x < featDim; x++)
                res[y] += (buffer[x] * coeff[x]);

            coeff += featDim;
        }

        feats += featDim;
        res += dim;
    }

    delete [] buffer;

    return dim;
}


void release(PCAModel **model){
    PCAModel *ptrModel = *model;
    if(*model == NULL)
        return;

    if(ptrModel->vmean != NULL)
        delete [] ptrModel->vmean;
    ptrModel->vmean = NULL;

    if(ptrModel->vstd != NULL)
        delete [] ptrModel->vstd;
    ptrModel->vstd = NULL;

    if(ptrModel->eigenValues != NULL)
        delete [] ptrModel->eigenValues;

    ptrModel->eigenValues = NULL;

    if(ptrModel->eigenVectors != NULL)
        delete [] ptrModel->eigenVectors;

    ptrModel->eigenVectors = NULL;

    delete *model;
    *model = NULL;
}


int save_pca_model(FILE *fout, PCAModel *model){
    if(fout == NULL){
        printf("Can't write model\n");
        return RET_FILE_ERROR;
    }

    int ret = 0;

    if(fwrite(&model->featDim, sizeof(int), 1, fout) != 1)
        return RET_IO_ERROR;

    if(fwrite(&model->dim, sizeof(int), 1, fout) != 1)
        return RET_IO_ERROR;

    if(fwrite(&model->flag, sizeof(int), 1, fout) != 1)
        return RET_IO_ERROR;

    if(fwrite(model->vmean, sizeof(float), model->featDim, fout) != model->featDim)
        return RET_IO_ERROR;

    if(model->flag == 1){
        if(fwrite(model->vstd, sizeof(float), model->featDim, fout) != model->featDim)
            return RET_IO_ERROR;
    }

    if(fwrite(model->eigenValues, sizeof(float), model->dim, fout) != model->dim)
        return RET_IO_ERROR;

    if(fwrite(model->eigenVectors, sizeof(float), model->featDim * model->dim, fout) != model->dim * model->featDim)
        return RET_IO_ERROR;

    return RET_OK;
}



int save(const char *filePath, PCAModel *model){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL){
        return 1;
    }

    int ret = save_pca_model(fout, model);

    fclose(fout);
    return ret;
}

int load_pca_model(FILE *fin, PCAModel **ptrModel){
    int featDim, dim, flag;

    if(fin == NULL || ptrModel == NULL){
        printf("Can't read model\n");
        return RET_FILE_ERROR;
    }

    int ret = 0;

    if(fread(&featDim, sizeof(int), 1, fin) != 1)
        return RET_IO_ERROR;

    if(fread(&dim, sizeof(int), 1, fin) != 1)
        return RET_IO_ERROR;

    if(fread(&flag, sizeof(int), 1, fin) != 1);

    PCAModel *model = create_pca_data(featDim, dim);

    model->flag = flag;

    if(fread(model->vmean, sizeof(float), model->featDim, fin) != model->featDim)
        return RET_IO_ERROR;

    if(model->flag == 1){
        if(fread(model->vstd, sizeof(float), model->featDim, fin) != model->featDim)
            return RET_IO_ERROR;
    }

    if(fread(model->eigenValues, sizeof(float), model->dim, fin) != model->dim)
        return RET_IO_ERROR;

    if(fread(model->eigenVectors, sizeof(float), model->featDim * model->dim, fin) != model->dim * model->featDim)
        return RET_IO_ERROR;

    *ptrModel = model;

    return RET_OK;
}


int load(const char *filePath, PCAModel **pcaModel){
    FILE *fin = fopen(filePath, "rb");

    if(fin == NULL)
        return 1;

    int ret = load_pca_model(fin, pcaModel);

    fclose(fin);

    return ret;
}




void pca_projection_feature(float *feats, int featDim, int dim, float *meanFeats, float *eigenMatrix, float *resFeats){
    int f, d;

    float *bufferFeats = new float[featDim];
    float *ptrMeans = meanFeats;
    float *ptrFeats = bufferFeats;

    memcpy(ptrFeats, feats, sizeof(float) * featDim);

    f = 0;
    for(; f <= featDim - 8; f += 8){
        ptrFeats[0] -= ptrMeans[0]; ptrFeats[1] -= ptrMeans[1];
        ptrFeats[2] -= ptrMeans[2]; ptrFeats[3] -= ptrMeans[3];

        ptrFeats[4] -= ptrMeans[4]; ptrFeats[5] -= ptrMeans[5];
        ptrFeats[6] -= ptrMeans[6]; ptrFeats[7] -= ptrMeans[7];

        ptrFeats += 8, ptrMeans += 8;
    }

    for(; f < featDim; f++){
        ptrFeats[0] -= ptrMeans[0];
        ptrFeats ++, ptrMeans ++;
    }

    for(d = 0; d < dim; d++){
        float t;
        float *eigen = eigenMatrix + d * featDim;

        ptrFeats = bufferFeats;

        t = 0;
        f = 0;
        for(; f <= featDim - 8; f += 8){
            t += ptrFeats[0] * eigen[0]; t += ptrFeats[1] * eigen[1];
            t += ptrFeats[2] * eigen[2]; t += ptrFeats[3] * eigen[3];

            t += ptrFeats[4] * eigen[4]; t += ptrFeats[5] * eigen[5];
            t += ptrFeats[6] * eigen[6]; t += ptrFeats[7] * eigen[7];

            ptrFeats += 8, eigen += 8;
        }

        for(; f < featDim; f++){
            t += ptrFeats[0] * eigen[0];

            ptrFeats ++, eigen ++;
        }

        resFeats[d] = t;
    }

    delete [] bufferFeats;
}

