#include "aligner.h"


PCAModel* create_pca_data(int featDim, int dim){
    PCAModel *model = new PCAModel;

    memset(model, 0, sizeof(PCAModel));

    model->vmean = new float[featDim];
    model->vstd = new float[featDim];

    model->eigenValues = new float[dim];
    model->eigenVectors = new float[featDim * dim];

    model->dim = dim;
    model->featDim = featDim;

    model->flag = 0;

    return model;
}


void denoise(float *feats, int ssize, int dim, PCAModel *model, float factor){
    int featDim = model->featDim;

    float *arr = new float[dim];
    float *field = model->eigenValues;

    assert(dim <= model->dim);

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

void denoise_shape(PCAModel *model, Shape &shape, Shape &meanShape){
    TranArgs arg;
    int ptsSize = meanShape.ptsSize;
    int ptsSize2 = ptsSize << 1;

    similarity_transform(shape, meanShape, arg);

    affine_shape(shape, shape, arg);

    for(int p = 0; p < ptsSize2; p++){
        shape.pts[p] = (shape.pts[p] - model->vmean[p]) / model->vstd[p];
    }

    denoise(shape.pts, 1, ptsSize2, model, 1.0f);

    for(int p = 0; p < ptsSize2; p++)
        shape.pts[p] = shape.pts[p] * model->vstd[p] + model->vmean[p];

    HU_SWAP(arg.cen1, arg.cen2, HPoint2f);
    arg.angle = -arg.angle;
    arg.scale = 1.0f / arg.scale;

    affine_shape(shape, shape, arg);
}


int projection_feature(float *feats, PCAModel *model, float *res){
    int featDim = model->featDim;
    int dim = model->dim;

    float *coeff = model->eigenVectors;
    float *vmean = model->vmean;

    for(int j = 0; j < featDim; j++)
        feats[j] -= vmean[j];

    memset(res, 0, sizeof(float) * dim);
    for(int y = 0; y < dim; y++){
        for(int x = 0; x < featDim; x++)
            res[y] += (feats[x] * coeff[x]);
        coeff += featDim;
    }

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


#define CELL_NUM    4
#define PYRAMID_SIZE 2


#define SEG_0 0
#define SEG_1 4
#define SEG_2 20
#define SEG_3 24
#define CELL_CX 12
#define CELL_CY 12
#define PATCH_RADIUS 13

typedef struct {
    uint8_t direct;
    uint16_t grad;
} GradPair;

GradPair gradTable[256][256];


static void init_grad_table(int gradDirect){
    for(int y = 0; y < 256; y++){
        for(int x = 0; x < 256; x++){
            if(y == 0 && x == 0){
                gradTable[0][0].direct = 0;
                gradTable[0][0].grad = 0.0f;
            }

            gradTable[y][x].direct = ((y - x + 0.0f) / (y + x + 0.0f) + 1) / 2.0f * (gradDirect - 1);
            gradTable[y][x].grad = sqrtf(y * y + x * x) + 0.5f;
        }
    }
}


static void extract_points_based_features(uint8_t *img, int width, int height, int stride, Shape &shape, PCAModel **models, int gradDirect, float *feats){
    int ptsSize = shape.ptsSize;
    float *pts = shape.pts;

    int LAYER_DIM = gradDirect * CELL_NUM;
    int BLOCK_DIM = gradDirect * PYRAMID_SIZE;
    int ONE_PT_DIM = LAYER_DIM * PYRAMID_SIZE;

    HRect rect = get_shape_rect(shape);
    int x0, y0, x1, y1;
    int bl, br, bt, bb;
    int w, h;
    int wt, ht;

    x0 = rect.x - PATCH_RADIUS;
    y0 = rect.y - PATCH_RADIUS;
    x1 = rect.x + rect.width + PATCH_RADIUS;
    y1 = rect.y + rect.height + PATCH_RADIUS;

    w = x1 - x0 + 1;
    h = y1 - y0 + 1;

    bl = br = bt = bb = 0;

    if(x0 < 0) {
        bl = -x0;
        x0 = 0;
    }

    if(y0 < 0){
        bt = -y0;
        y0 = 0;
    }

    if(x1 >= width){
        br = x1 - width + 1;
        x1 = width - 1;
    }

    if(y1 >= height){
        bb = y1 - height + 1;
        y1 = height - 1;
    }

    wt = x1 - x0 + 1;
    ht = y1 - y0 + 1;

    GradPair *pairs = new GradPair[w * h], *pPairs;

    memset(pairs, 0, sizeof(GradPair) * w * h);

    img += y0 * stride + x0;
    pPairs = pairs + bt * w + bl;
    for(int y = 0; y < ht; y++){
        for(int x = 0; x < wt; x++){
            int dx = abs(img[x] - img[x + 1]);
            int dy = abs(img[x] - img[x + stride]);

            pPairs[y * w + x] = gradTable[dx][dy];
        }

        img += stride;
    }

    memset(feats, 0, sizeof(float) * ONE_PT_DIM * ptsSize);

    for(int i = 0; i < ptsSize; i++){
        int lx, ty;
        float *ptr1, *ptr2, *ptr3;

        GradPair *ptrPair;

        float onefeats[100];

        memset(onefeats, 0, sizeof(float) * 100);

        lx = int(pts[i] - 11.5f) - x0;
        ty = int(pts[i + ptsSize] - 11.5f) - y0;

        ptr1 = onefeats;
        ptr2 = ptr1 + gradDirect;

        ptrPair = pairs + ty * w + lx;
        for(int y = SEG_0; y < SEG_1; y ++){
            int m = (y >= CELL_CY) << 1;

            for(int x = SEG_0; x < SEG_3; x++){
                GradPair pair = ptrPair[x];
                int id = (m + (x >= CELL_CX)) * BLOCK_DIM + pair.direct;
                ptr1[id] += pair.grad;
            }

            ptrPair += w;
        }

        for(int y = SEG_1; y < SEG_2; y ++){
            int m = (y >= CELL_CY) << 1;
            for(int x = SEG_0; x < SEG_1; x++){
                GradPair pair = ptrPair[x];
                int id = (m + (x >= CELL_CX)) * BLOCK_DIM + pair.direct;
                ptr1[id] += pair.grad;
            }

            for(int x = SEG_1; x < SEG_2; x++){
                GradPair pair = ptrPair[x];
                int id = (m + (x >= CELL_CX)) * BLOCK_DIM + pair.direct;
                ptr1[id] += pair.grad;
                ptr2[id] += pair.grad;
            }

            for(int x = SEG_2; x < SEG_3; x++){
                GradPair pair = ptrPair[x];
                int id = (m + (x >= CELL_CX)) * BLOCK_DIM + pair.direct;
                ptr1[id] += pair.grad;
            }
            ptrPair += w;
        }

        for(int y = SEG_2; y < SEG_3; y++){
            int m = (y >= CELL_CY) << 1;

            for(int x = SEG_0; x < SEG_3; x++){
                GradPair pair = ptrPair[x];
                int id = (m + (x >= CELL_CX)) * BLOCK_DIM + pair.direct;
                ptr1[id] += pair.grad;
            }

            ptrPair += w;
        }

        float *ptra = onefeats;
        float *ptrb = ptra + BLOCK_DIM;
        float *ptrc = ptrb + BLOCK_DIM;
        float *ptrd = ptrc + BLOCK_DIM;

        int j = 0;
        //*
#ifdef __ARM_NEON
        for(; j <= BLOCK_DIM - 4; j += 4){
            float32x4_t f;
            float32x4_t a, b, c, d;
            float32x4_t a2, b2, c2, d2;

            a = vld1q_f32(ptra + j);
            b = vld1q_f32(ptrb + j);
            c = vld1q_f32(ptrc + j);
            d = vld1q_f32(ptrd + j);

            a2 = vmulq_f32(a, a);
            b2 = vmulq_f32(b, b);
            c2 = vmulq_f32(c, c);
            d2 = vmulq_f32(d, d);

            f = vmulq_f32(vrsqrteq_f32(vaddq_f32(vaddq_f32(vaddq_f32(a2, b2), vaddq_f32(c2, d2)), vdupq_n_f32(FLT_EPSILON))), vdupq_n_f32(2.0f));

            a = vmulq_f32(f, a);
            b = vmulq_f32(f, b);
            c = vmulq_f32(f, c);
            d = vmulq_f32(f, d);

            vst1q_f32(ptra + j, a);
            vst1q_f32(ptrb + j, b);
            vst1q_f32(ptrc + j, c);
            vst1q_f32(ptrd + j, d);
        }
#endif
        // */

        for(; j < BLOCK_DIM; j++){
            float factor;

            factor = ptra[j] * ptra[j] + ptrb[j] * ptrb[j] + ptrc[j] * ptrc[j] + ptrd[j] * ptrd[j];

            factor = 2.0f / (sqrtf(factor) + 0.000001f);

            ptra[j] *= factor; ptrb[j] *= factor; ptrc[j] *= factor; ptrd[j] *= factor;
        }

        feats += projection_feature(onefeats, models[i], feats);
    }

    delete [] pairs;
}


int project_points_features(float *feats, int oneDim, PCAModel **ptsPCA, int ptsSize, float *res){
    int regDim = 0;

    for(int i = 0; i < ptsSize; i++){
        PCAModel *model = ptsPCA[i];

        int featDim = model->featDim;
        int dim = model->dim;

        float *coeff = model->eigenVectors;
        float *vmean = model->vmean;

        assert(featDim == oneDim);

        for(int j = 0; j < featDim; j++)
            feats[j] -= vmean[j];

        memset(res, 0, sizeof(float) * dim);
        for(int y = 0; y < dim; y++){
            for(int x = 0; x < featDim; x++)
                res[y] += (feats[x] * coeff[x]);
            coeff += featDim;
        }

        feats += featDim;
        res += dim;
        regDim += dim;
    }

    return regDim;
}
#define FEAT_FIX_Q 6
#define FEAT_FIX_ONE (1 << FEAT_FIX_Q)

#define COEFF_FIX_Q 9
#define COEFF_FIX_ONE (1 << COEFF_FIX_Q)

double DELTA_FACTOR = 1.0f / (FEAT_FIX_ONE * COEFF_FIX_ONE);


static inline void vector_mul_add(int16_t t, int *ptrShape, int16_t *ptrCoeff, int ptsSize2){
    int p = 0;

    for(; p <= ptsSize2 - 8; p += 8){
        ptrShape[0] += t * ptrCoeff[0];
        ptrShape[1] += t * ptrCoeff[1];
        ptrShape[2] += t * ptrCoeff[2];
        ptrShape[3] += t * ptrCoeff[3];

        ptrShape[4] += t * ptrCoeff[4];
        ptrShape[5] += t * ptrCoeff[5];
        ptrShape[6] += t * ptrCoeff[6];
        ptrShape[7] += t * ptrCoeff[7];

        ptrShape += 8;
        ptrCoeff += 8;
    }

    for(; p < ptsSize2; p++, ptrShape++, ptrCoeff ++)
        ptrShape[0] += t * ptrCoeff[0];

}


static void sdm_predict(Aligner *aligner, int s, float *feat, float *shape){
    PCAModel *rsdlPCA = aligner->rsdlPCA[s];
    int16_t *coeff = aligner->coeffs[s];

    int ptsSize2 = aligner->ptsSize << 1;
    int featDim = aligner->featDim;
    int dim = rsdlPCA->dim;

    float *meanResidual = rsdlPCA->vmean;
    float *eigenMatrix = rsdlPCA->eigenVectors;

    int i = 0;

    int delta[202];

    assert(ptsSize2 = rsdlPCA->featDim);

    memset(delta, 0, sizeof(int) * dim);

    for(i = 0; i <= featDim - 8; i += 8, feat += 8){
        int16_t t[] = {
            int16_t(feat[0] * FEAT_FIX_ONE), int16_t(feat[1] * FEAT_FIX_ONE),
            int16_t(feat[2] * FEAT_FIX_ONE), int16_t(feat[3] * FEAT_FIX_ONE),
            int16_t(feat[4] * FEAT_FIX_ONE), int16_t(feat[5] * FEAT_FIX_ONE),
            int16_t(feat[6] * FEAT_FIX_ONE), int16_t(feat[7] * FEAT_FIX_ONE)
        };

        vector_mul_add(t[0], delta, coeff, dim); coeff += dim;
        vector_mul_add(t[1], delta, coeff, dim); coeff += dim;
        vector_mul_add(t[2], delta, coeff, dim); coeff += dim;
        vector_mul_add(t[3], delta, coeff, dim); coeff += dim;

        vector_mul_add(t[4], delta, coeff, dim); coeff += dim;
        vector_mul_add(t[5], delta, coeff, dim); coeff += dim;
        vector_mul_add(t[6], delta, coeff, dim); coeff += dim;
        vector_mul_add(t[7], delta, coeff, dim); coeff += dim;
    }

    for(; i < featDim; i++, coeff += dim, feat++)
        vector_mul_add(int(feat[0]), delta, coeff, dim);

    for(i = 0; i < dim; i++){
        float t = delta[i] * DELTA_FACTOR;

        for(int j = 0; j < ptsSize2; j++)
            shape[j] += t * eigenMatrix[j];

        eigenMatrix += ptsSize2;
    }

    for(int j = 0; j < ptsSize2; j++)
        shape[j] += meanResidual[j];
}

void smooth_shape(Shape &baseShape, Shape &resShape);


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, Shape &curShape){
    if(aligner->stage == 0) return;

    int ptsSize = aligner->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int featDim = aligner->featDim;

    Shape baseShape;
    TranArgs arg;
    HRect rect, srect;

    uint8_t *patch;
    uint8_t src[SAMPLE_IMAGE_SIZE2 * SAMPLE_IMAGE_SIZE2 * 2];
    uint8_t *srcBk = src + SAMPLE_IMAGE_SIZE2 * SAMPLE_IMAGE_SIZE2;
    float scale;

    assert(aligner != NULL && img != NULL);

    assert(featDim < 2500);

    srect = get_shape_rect(curShape);

    rect.x = srect.x - (srect.width  >> 1);
    rect.y = srect.y - (srect.height >> 1);
    rect.width  = srect.width * 2 + 1;
    rect.height = srect.height * 2 + 1;

    patch = new uint8_t[rect.width * rect.height];

    extract_area_from_image(img, width, height, stride, patch, rect);

    scale = SAMPLE_IMAGE_SIZE2 / (float)rect.width;

    resizer_bilinear_gray(patch, rect.width, rect.height, rect.width, src, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2);

    delete [] patch;

    for(int p = 0; p < ptsSize; p++){
        curShape.pts[p] = (curShape.pts[p] - rect.x) * scale;
        curShape.pts[p + ptsSize] = (curShape.pts[p + ptsSize] - rect.y) * scale;
    }

    memcpy(srcBk, src, sizeof(uint8_t) * SAMPLE_IMAGE_SIZE2 * SAMPLE_IMAGE_SIZE2);
    baseShape = curShape;

    for(int s = 0; s < aligner->stage; s++){
        float feats[2500];

        memcpy(src, srcBk, sizeof(uint8_t) * SAMPLE_IMAGE_SIZE2 * SAMPLE_IMAGE_SIZE2);

        similarity_transform(baseShape, aligner->meanShape, arg);
        arg.cen2.x = SAMPLE_IMAGE_SIZE2 >> 1;
        arg.cen2.y = SAMPLE_IMAGE_SIZE2 >> 1;

        affine_sample(src, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2, baseShape, arg);

        extract_points_based_features(src, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2, baseShape, aligner->pointsPCA, aligner->gradDirect, feats);

        sdm_predict(aligner, s, feats, baseShape.pts);

        if(!aligner->isAlign)
            denoise_shape(aligner->shapePCA, baseShape, aligner->meanShape);

        arg.scale = 1.0f / arg.scale;
        arg.angle = -arg.angle;

        HU_SWAP(arg.cen1, arg.cen2, HPoint2f);
        affine_shape(baseShape, baseShape, arg);

#ifdef __ARM_NEON
        //__android_log_print(ANDROID_LOG_INFO, "jni", "%f %f %f", t1, t2, t3);

#endif

    }

    smooth_shape(curShape, baseShape);
    //show_shape(src, winSize, winSize, winSize, curShape, "cur");

    scale = 1.0f / scale;
    for(int p = 0; p < ptsSize; p++){
        curShape.pts[p] = baseShape.pts[p] * scale + rect.x;
        curShape.pts[p + ptsSize] = baseShape.pts[p + ptsSize] * scale + rect.y;
    }
}


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, HRect &srect, Shape &curShape){
    int ptsSize = aligner->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int featDim = aligner->featDim;
    TranArgs arg;
    HRect rect;
    int winSize = SAMPLE_IMAGE_SIZE * 2 + 1;
    float scale;

    float feats[2500];

    assert(featDim < 2500);

    assert(aligner != NULL && img != NULL);

    rect.x = srect.x - (srect.width  >> 1);
    rect.y = srect.y - (srect.height >> 1);
    rect.width  = srect.width << 1;
    rect.height = srect.height << 1;

    uint8_t *patch = new uint8_t[rect.width * rect.height];
    uint8_t src[(SAMPLE_IMAGE_SIZE * 2 + 1) * (SAMPLE_IMAGE_SIZE * 2 + 1) * 2];
    uint8_t *srcBk = src + (SAMPLE_IMAGE_SIZE * 2 + 1) * (SAMPLE_IMAGE_SIZE * 2 + 1);

    extract_area_from_image(img, width, height, stride, patch, rect);

    scale = (float)rect.width / winSize;

    resizer_bilinear_gray(patch, rect.width, rect.height, rect.width, src, winSize, winSize, winSize);
    delete [] patch;

    curShape = aligner->meanShape;

    memcpy(srcBk, src, sizeof(uint8_t) * winSize * winSize);

    for(int p = 0; p < ptsSize2; p++)
        curShape.pts[p] += SAMPLE_IMAGE_SIZE;

    for(int s = 0; s < aligner->stage; s++){
        memcpy(src, srcBk, sizeof(uint8_t) * winSize * winSize);

        similarity_transform(curShape, aligner->meanShape, arg);

        arg.cen2.x = winSize >> 1;
        arg.cen2.y = winSize >> 1;

        affine_sample(src, winSize, winSize, winSize, curShape, arg);

        extract_points_based_features(src, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2, SAMPLE_IMAGE_SIZE2, curShape, aligner->pointsPCA, aligner->gradDirect, feats);

        sdm_predict(aligner, s, feats, curShape.pts);

        arg.scale = 1.0f / arg.scale;
        arg.angle = -arg.angle;

        HU_SWAP(arg.cen1, arg.cen2, HPoint2f);
        affine_shape(curShape, curShape, arg);
    }

    for(int p = 0; p < ptsSize; p++){
        curShape.pts[p] = curShape.pts[p] * scale + rect.x;
        curShape.pts[p + ptsSize] = curShape.pts[p + ptsSize] * scale + rect.y;
    }
}


int load_coeff(int16_t *coeff, int rows, int cols, FILE *fin){
    if(fin == NULL || coeff == NULL)
        return 1;

    int ret;
    uint8_t *arr = NULL;
    int16_t *ptr = NULL;

    int t;
    ret = fread(&t, sizeof(int), 1, fin); assert(ret == 1);
    assert(t == rows);
    ret = fread(&t, sizeof(int), 1, fin); assert(ret == 1);
    assert(t == cols);

    ptr = coeff;

    arr = (uint8_t*)malloc(sizeof(uint8_t) * cols);

    for(int y = 0; y < rows; y++){
        float minv = FLT_MAX, maxv = -FLT_MAX;
        float step = 0.0f;

        ret = fread(&minv, sizeof(float), 1, fin); assert(ret == 1);
        ret = fread(&step, sizeof(float), 1, fin); assert(ret == 1);

        ret = fread(arr, sizeof(uint8_t), cols, fin); assert(ret == cols);

        for(int x = 0; x < cols; x++)
            ptr[x] = (arr[x] * step + minv) * COEFF_FIX_ONE;

        ptr += cols;
    }

    free(arr);

    return 0;
}


int load(const char *filePath, Aligner **raligner){
    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL) {
        return 1;
    }

    int ret;

    Aligner *aligner = new Aligner;

    memset(aligner, 0, sizeof(Aligner));


    ret = fread(&aligner->stage, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->isAlign, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->ptsSize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->featDim, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->gradDirect, sizeof(int), 1, fin); assert(ret == 1);

    aligner->oneDim = aligner->gradDirect * CELL_NUM * PYRAMID_SIZE;

    ret = fread(&aligner->meanShape, sizeof(Shape), 1, fin); assert(ret == 1);

    aligner->pointsPCA = new PCAModel*[aligner->ptsSize];
    aligner->coeffs = new int16_t*[aligner->stage];
    aligner->rsdlPCA = new PCAModel*[aligner->stage];

    for(int i = 0; i < aligner->ptsSize; i++){
        ret = load_pca_model(fin, &aligner->pointsPCA[i]); assert(ret == 0);
    }

    for(int i = 0; i < aligner->stage; i++){
        ret = load_pca_model(fin, &aligner->rsdlPCA[i]); assert(ret == 0);

        aligner->coeffs[i] = new int16_t[aligner->featDim * aligner->rsdlPCA[i]->dim];

        ret = load_coeff(aligner->coeffs[i], aligner->featDim, aligner->rsdlPCA[i]->dim, fin); assert(ret == 0);
    }

    ret = load_pca_model(fin, &aligner->shapePCA); assert(ret == 0);

    fclose(fin);

    init_grad_table(aligner->gradDirect);

    *raligner = aligner;


    return 0;
}


void release_data(Aligner *aligner){
    if(aligner == NULL)
        return;

    release(&aligner->shapePCA);

    if(aligner->coeffs != NULL){
        for(int i = 0; i < aligner->stage; i++){
            delete [] aligner->coeffs[i];
            release(&aligner->rsdlPCA[i]);
        }

        for(int i = 0; i <aligner->ptsSize; i++)
            release(&aligner->pointsPCA[i]);

        delete [] aligner->coeffs;
        delete [] aligner->rsdlPCA;
        delete [] aligner->pointsPCA;
    }

    aligner->coeffs = NULL;
    aligner->rsdlPCA = NULL;
    aligner->shapePCA = NULL;
    aligner->stage = 0;

}

void release(Aligner **aligner){
    if(*aligner == NULL)
        return ;

    release_data(*aligner);

    delete *aligner;
    *aligner = NULL;
}



void smooth_shape(Shape &baseShape, Shape &resShape){
    float residual[202];

    float dx = 0, dy = 0;
    float *ptr;

    int ptsSize = baseShape.ptsSize;
    float *lastShape = baseShape.pts;
    float *curShape = resShape.pts;

    for(int i = 0; i < ptsSize; i++){
        residual[i] = lastShape[i] - curShape[i];
        dx += residual[i];

        residual[i + ptsSize] = lastShape[i + ptsSize] - curShape[i + ptsSize];
        dy += residual[i + ptsSize];
    }

    dx /= ptsSize;
    dy /= ptsSize;

    //global smooth
    float rate = 1;
    if(fabs(dx) < 0.16)
        rate = 0;
    else if(fabs(dx) < 0.4)
        rate = 0.6;
    else if(fabs(dx) < 0.64)
        rate = 0.8;
    else if(fabs(dx) < 0.8)
        rate = 0.9;

    ptr = residual;
    for(int i = 0; i < ptsSize; i++){
        ptr[i] *= rate;
        float value = fabs(ptr[i]);
        if (value > 2.4){}
        else if(value > 1.6)
            ptr[i] *= 0.9;
        else if(value > 0.8)
            ptr[i] *= 0.8;
        else if(value > 0.4)
            ptr[i] *= 0.6;
        else if(value > 0.16)
            ptr[i] *= 0.4;
        else
            ptr[i] = 0;
    }

    rate = 1;
    if(fabs(dy) < 0.16)
        rate = 0;
    else if(fabs(dy) < 0.4)
        rate = 0.6;
    else if(fabs(dy) < 0.64)
        rate = 0.8;
    else if(fabs(dy) < 0.8)
        rate = 0.9;

    ptr = residual + ptsSize;
    for(int i = 0; i < ptsSize; i++){
        ptr[i] *= rate;
        float value = fabs(ptr[i]);
        if (value > 2.4){}
        else if(value > 1.6)
            ptr[i] *= 0.9;
        else if(value > 0.8)
            ptr[i] *= 0.8;
        else if(value > 0.4)
            ptr[i] *= 0.6;
        else if(value > 0.16)
            ptr[i] *= 0.4;
        else
            ptr[i] = 0;
    }


    for(int i = 0; i < ptsSize; i++){
        curShape[i] = lastShape[i] - residual[i];
        curShape[i + ptsSize] = lastShape[i + ptsSize] - residual[i + ptsSize];
    }
}


