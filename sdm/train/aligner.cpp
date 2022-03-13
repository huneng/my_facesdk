#include "aligner.h"

#define GRAD_DIRECT  5
#define PYRAMID_SIZE 2

#define TIMES 8
#define MIRROR 1

#define CELL_NUM   4

#define FEATURE_FILE "log/traindata.bin"
#define FEAT_PCA_FILE "log/feat_pca.dat"

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


#define SEG_0 0
#define SEG_1 4
#define SEG_2 20
#define SEG_3 24
#define CELL_CX 12
#define CELL_CY 12
#define PATCH_RADIUS 13

static void extract_points_based_features(uint8_t *img, int width, int height, int stride, Shape &shape, int gradDirect, float *feats){
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

        lx = int(pts[i] - 11.5f) - x0;
        ty = int(pts[i + ptsSize] - 11.5f) - y0;

        ptr1 = feats;
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

        float *ptra = feats;
        float *ptrb = ptra + BLOCK_DIM;
        float *ptrc = ptrb + BLOCK_DIM;
        float *ptrd = ptrc + BLOCK_DIM;

        int j = 0;
        for(; j < BLOCK_DIM; j++){
            float factor;

            factor = ptra[j] * ptra[j] + ptrb[j] * ptrb[j] + ptrc[j] * ptrc[j] + ptrd[j] * ptrd[j];

            factor = 2.0f / (sqrtf(factor) + 0.000001f);

            ptra[j] *= factor; ptrb[j] *= factor; ptrc[j] *= factor; ptrd[j] *= factor;
        }

        feats += ONE_PT_DIM;
    }

    delete [] pairs;
}


static float feature_distance(float *feats1, float *feats2, int size){
    float value = 0;

    for(int i = 0; i < size; i++)
        value += fabs(feats1[i] - feats2[i]);

    return value / size;
}


static void get_shape_center(float *shape, int ptsSize, HPoint2f &center){
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    for(int i = 0; i < ptsSize; i++){
        float x = shape[i];
        float y = shape[i + ptsSize];

        minx = HU_MIN(minx, x);
        maxx = HU_MAX(maxx, x);

        miny = HU_MIN(miny, y);
        maxy = HU_MAX(maxy, y);
    }

    center.x = 0.5f * (minx + maxx);
    center.y = 0.5f * (miny + maxy);
}


void pca_projection_feature(float *feats, int featDim, int dim, float *meanFeats, float *eigenMatrix, float *resFeats){
    int f, d;

    float *ptrFeats = feats;
    float *ptrMeans = meanFeats;

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

        ptrFeats = feats;

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
}


int project_points_features(float *feats, int oneDim, PCAModel **ptsPCA, int ptsSize, float *res);


static void generate_features_for_align(Aligner *aligner, SampleSet *set, int times, float *feats, float *rsdls){
    int ptsSize = aligner->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int featDim = aligner->featDim;
    int ssize = set->ssize;
    int WINW = set->WINW;
    int WINH = set->WINH;
    int sq = WINW * WINH;

    cv::RNG rng(cv::getTickCount());

    float ANGLE_80 = 80 / 180.0f * HU_PI;

    Shape meanShape = aligner->meanShape;

    int finished = 0;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < ssize; i++){
        uint8_t *img = new uint8_t[sq];

        float *featsBuffer = new float[times * featDim];
        float *curFeat = new float[featDim];
        float *oriFeat = new float[aligner->oneDim * aligner-> ptsSize];

        Shape curShape, oriShape;
        TranArgs arg;

        Sample *sample = set->samples[i];

        int count = 0;

        float RAND_LEN = 0.1 * SAMPLE_IMAGE_SIZE;

        while(count < times){
            int id = i * times + count;

            float *ptrFeat = feats + id * featDim;
            float *ptrRsdl = rsdls + id * ptsSize2;

            int flag;

            HPoint2f center;
            HRect rect;

            memcpy(img, sample->img, sizeof(uint8_t) * sq);
            oriShape = sample->oriShape;

            similarity_transform(oriShape, meanShape, arg);

            arg.scale *= rng.uniform(0.85, 1.15);
            arg.angle += rng.uniform(-HU_PI / 4.0f, HU_PI / 4.0f);
            arg.cen2 = arg.cen1;

            affine_sample(img, WINW, WINH, WINW, oriShape, arg);
            transform_image(img, WINW, WINH, WINW);

            get_shape_center(oriShape.pts, oriShape.ptsSize, center);

            rect.x = center.x - (SAMPLE_IMAGE_SIZE >> 1) + rng.uniform(-RAND_LEN, RAND_LEN);
            rect.y = center.y - (SAMPLE_IMAGE_SIZE >> 1) + rng.uniform(-RAND_LEN, RAND_LEN);
            rect.width = SAMPLE_IMAGE_SIZE;
            rect.height = SAMPLE_IMAGE_SIZE;

            //show_rect(img, WINW, WINH, WINW, rect, "cur");
            predict(aligner, img, WINW, WINH, WINW, rect, curShape);
            //show_shape(img, WINW, WINH, WINW, curShape, "res");

            similarity_transform(curShape, meanShape, arg);
            arg.cen2.x = WINW >> 1;
            arg.cen2.y = WINH >> 1;

            affine_sample(img, WINW, WINH, WINW, curShape, arg);
            affine_shape(oriShape, oriShape, arg);

            extract_points_based_features(img, WINW, WINH, WINW, curShape, aligner->gradDirect, oriFeat);
            project_points_features(oriFeat, aligner->oneDim, aligner->pointsPCA, aligner->ptsSize, curFeat);

            flag = 0;
            for(int fid = 0; fid < count; fid++){
                if(feature_distance(curFeat, featsBuffer + fid * featDim, featDim) < 0.1f) {
                    flag = 1;
                    break;
                }
            }

            if(flag) continue;

            memcpy(featsBuffer + count * featDim, curFeat, sizeof(float) * featDim);
            count++;

            memcpy(ptrFeat, curFeat, sizeof(float) * featDim);

            for(int p = 0; p < ptsSize2; p++)
                ptrRsdl[p] = oriShape.pts[p] - curShape.pts[p];
        }

#pragma omp cirtical
        {
            finished++;
            printf("%d\r", finished), fflush(stdout);
        }

        delete [] img;
        delete [] featsBuffer;
        delete [] curFeat;
        delete [] oriFeat;
    }
}


static void generate_features_for_track(Aligner *aligner, SampleSet *set, int times, float *feats, float *rsdls){
    int ptsSize = aligner->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int featDim = aligner->featDim;
    int ssize = set->ssize;
    int WINW = set->WINW;
    int WINH = set->WINH;
    int sq = WINW * WINH;

    cv::RNG rng(cv::getTickCount());

    Shape meanShape = aligner->meanShape;


    int finished = 0;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < ssize; i++){
        uint8_t *img = new uint8_t[sq];

        float *featsBuffer = new float[times * featDim];
        float *curFeat = new float[featDim];
        float *oriFeat = new float[aligner->oneDim * aligner->ptsSize];

        Shape curShape, oriShape;
        TranArgs arg;

        Sample *sample = set->samples[i];

        int RAND_LEN = 0.15f * SAMPLE_IMAGE_SIZE;
        int NOISE_LEN = 0.1f * SAMPLE_IMAGE_SIZE;

        int count = 0;
        while(count < times){
            int id = i * times + count;

            float *ptrFeat = feats + id * featDim;
            float *ptrRsdl = rsdls + id * ptsSize2;

            int flag = 0, dx, dy;

            memcpy(img, sample->img, sizeof(uint8_t) * sq);
            oriShape = sample->oriShape;

            similarity_transform(oriShape, meanShape, arg);

            arg.scale *= rng.uniform(0.8, 1.2);
            arg.angle += rng.uniform(-HU_PI / 3, HU_PI / 3);
            arg.cen2 = arg.cen1;

            affine_sample(img, WINW, WINH, WINW, oriShape, arg);
            transform_image(img, WINW, WINH, WINW);

            curShape = oriShape;

            dx = rng.uniform(-RAND_LEN, RAND_LEN);
            dy = rng.uniform(-RAND_LEN, RAND_LEN);

            for(int p = 0; p < curShape.ptsSize; p++){
                curShape.pts[p] += (rng.uniform(-NOISE_LEN, NOISE_LEN) + dx);
                curShape.pts[p + ptsSize] += (rng.uniform(-NOISE_LEN, NOISE_LEN) + dy);
            }

            denoise_shape(aligner->shapePCA, curShape, aligner->meanShape);

            get_shape_center(curShape.pts, curShape.ptsSize, arg.cen1);
            arg.cen2 = arg.cen1;
            arg.angle = rng.uniform(-HU_PI / 9, HU_PI / 9);
            arg.scale = rng.uniform(0.9f, 1.1f);
            affine_shape(curShape, curShape, arg);

            //show_shape(img, WINW, WINH, WINW, curShape, "cur");
            predict(aligner, img, WINW, WINH, WINW, curShape);
            //show_shape(img, WINW, WINH, WINW, curShape, "cur");

            similarity_transform(curShape, meanShape, arg);
            arg.cen2.x = WINW >> 1;
            arg.cen2.y = WINH >> 1;
            affine_sample(img, WINW, WINH, WINW, curShape, arg);
            affine_shape(oriShape, oriShape, arg);
            //show_shape(img, WINW, WINH, WINW, oriShape, "ori");

            extract_points_based_features(img, WINW, WINH, WINW, curShape, aligner->gradDirect, oriFeat);
            project_points_features(oriFeat, aligner->oneDim, aligner->pointsPCA, aligner->ptsSize, curFeat);

            for(int fid = 0; fid < count; fid++){
                if(feature_distance(curFeat, featsBuffer + fid * featDim, featDim) < 0.1f){
                    flag = 1;
                    break;
                }
            }

            if(flag == 1) continue;

            memcpy(featsBuffer + count * featDim, curFeat, sizeof(float) * featDim);
            count++;

            memcpy(ptrFeat, curFeat, sizeof(float) * featDim);

            for(int p = 0; p < ptsSize2; p++)
                ptrRsdl[p] = oriShape.pts[p] - curShape.pts[p];
        }

#pragma omp cirtical
        {
            finished++;
            printf("%d\r", finished), fflush(stdout);
        }

        delete [] img;
        delete [] featsBuffer;
        delete [] curFeat;
        delete [] oriFeat;
    }
}


#define SOLVE_TYPE double

int solve_linear_equation(const char *filePath, float *XMat){
    FILE *fin = fopen(filePath, "rb");
    int ssize, featDim, ptsSize2, ret;
    float *buffer;


    int rows, cols, lda, ldb, nrhs;
    int maxMN, minMN, *iwork, rank, lwork, info, NLVL;
    SOLVE_TYPE *A, *B, *S, rcond, *work, wkopt;

    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    ret = fread(&ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&featDim, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&ptsSize2, sizeof(int), 1, fin); assert(ret == 1);


    rows = ssize;
    cols = featDim;

    lda = ssize;
    nrhs = ptsSize2;
    ldb = HU_MAX(rows, ssize);

    maxMN = HU_MAX(rows, cols);
    minMN = HU_MIN(rows, cols);

    buffer = (float*)malloc(sizeof(float) * featDim); assert(buffer != NULL);

    A = (SOLVE_TYPE*) malloc(sizeof(SOLVE_TYPE) * lda * cols); assert(A != NULL);
    B = (SOLVE_TYPE*) malloc(sizeof(SOLVE_TYPE) * nrhs * ldb); assert(B != NULL);
    S = (SOLVE_TYPE*) malloc(sizeof(SOLVE_TYPE) * minMN); assert(B != NULL);

    for(int y = 0; y < ssize; y++){
        ret = fread(buffer, sizeof(float), featDim, fin); assert(ret == featDim);
        for(int x = 0; x < featDim; x++)
            A[x * lda + y] = buffer[x];
    }

    for(int y = 0; y < ssize; y++){
        ret = fread(buffer, sizeof(float), ptsSize2, fin); assert(ret == ptsSize2);
        for(int x = 0; x < ptsSize2; x++)
            B[x * ldb + y] = buffer[x];
    }

    fclose(fin);

    NLVL = int( log2(minMN / (25 + 1.0)) + 2 );
    NLVL = HU_MAX(0, NLVL);
    iwork = new int[3 * minMN * NLVL + 11 * minMN];

    rcond = -1.0;
    lwork = -1;
    dgelsd_( &rows, &cols, &nrhs, A, &lda, B, &ldb, S, &rcond, &rank, &wkopt, &lwork, iwork, &info );

    lwork = (int)wkopt;
    work = (SOLVE_TYPE*)malloc(sizeof(SOLVE_TYPE) * lwork);

    dgelsd_( &rows, &cols, &nrhs, A, &lda, B, &ldb, S, &rcond, &rank, work, &lwork, iwork, &info );

    if(info > 0){
        printf( "The algorithm computing SVD failed to converge;\n" );
        printf( "The least squares solution could not be computed.\n" );
        exit(1);
    }

    for(int y = 0; y < featDim; y++)
        for(int x = 0; x < ptsSize2; x++)
            XMat[y * ptsSize2 + x] = B[x * ldb + y];

    free(A);
    free(B);
    free(S);
    free(work);
    free(iwork);

    free(buffer);
}


int solve_linear_equation_liblinear(const char *filePath, float *XMat){
    int ssize, featDim, ptsSize2, ret;
    float *buffer;

    FILE *fin = fopen(filePath, "rb");

    assert(fin != NULL);

    ret = fread(&ssize, sizeof(int), 1, fin);
    ret = fread(&featDim, sizeof(int), 1, fin);
    ret = fread(&ptsSize2, sizeof(int), 1, fin);

    buffer = (float*)malloc(sizeof(float) * featDim);

    struct problem   *prob   = new struct problem[ptsSize2];
    struct parameter *params = new struct parameter[ptsSize2];

    struct feature_node **nodes = new struct feature_node*[ssize];
    double **R = new double*[ptsSize2];

    struct model **models = new struct model*[ptsSize2];


    for(int i = 0; i < ssize; i++){
        nodes[i] = new struct feature_node[featDim + 1];

        ret = fread(buffer, sizeof(float), featDim, fin); assert(ret == featDim);

        for(int j = 0; j < featDim; j++){
            nodes[i][j].index = j + 1;
            nodes[i][j].value = buffer[j];
        }

        nodes[i][featDim].index = -1;
        nodes[i][featDim].value = -1;
    }

    for(int i = 0; i < ptsSize2; i++){
        R[i] = new double[ssize];

        prob[i].l = ssize;
        prob[i].n = featDim;
        prob[i].bias = -1;
        prob[i].x = nodes;
        prob[i].y = R[i];

        memset(params + i, 0, sizeof(struct parameter));

        params[i].solver_type =  L2R_L2LOSS_SVR_DUAL;
        params[i].C = 1.0 / ssize;
        params[i].p = 0.1;
        params[i].eps = 0.001;
    }

    for(int j = 0; j < ssize; j++){
        ret = fread(buffer, sizeof(float), ptsSize2, fin); assert(ret == ptsSize2);

        for(int i = 0; i < ptsSize2; i++)
            R[i][j] =  buffer[i];
    }

    delete [] buffer;

    fclose(fin);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < ptsSize2; i++){
        models[i] = train(prob + i, params + i);
        destroy_param(params + i);
    }


    for(int y = 0; y < featDim; y++){
        for(int x = 0; x < ptsSize2; x++)
            XMat[y * ptsSize2 + x] = models[x]->w[y];
    }

    write_matrix(XMat, ptsSize2, featDim, 1, "log/coeff.txt");

    for(int i = 0; i < ptsSize2; i++){
        free_model_content(models[i]);

        delete models[i];
        delete [] R[i];
    }

    for(int i = 0; i < ssize; i++){
        delete [] nodes[i];
        nodes[i] = NULL;
    }

    delete [] prob;
    delete [] params;

    delete [] nodes;
    delete [] models;
    delete [] R;
}



void mean_residual(float *rsdls, int ssize, int ptsSize){
    double sum = 0.0f;
    int ptsSize2 = ptsSize << 1;

    float factor = 100.0f / (SAMPLE_IMAGE_SIZE * ptsSize);

    float counts[11];

    memset(counts, 0, sizeof(float) * 11);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int y = 0; y < ssize; y++){
        float dist = 0;
        float *ptr = rsdls + y * ptsSize2;

        int id = 0;

        for(int x = 0; x < ptsSize; x++)
            dist += sqrtf(ptr[x] * ptr[x] + ptr[x + ptsSize] * ptr[x + ptsSize]);

        dist *= factor;

        id = (int)dist;
        id = HU_MIN(id, 10);

#pragma omp critical
        {
            counts[id] ++;
            sum += dist;
        }
    }


    for(int i = 0; i < 11; i++){
        counts[i] = counts[i] * 100.0 / ssize;
        if(i > 0)
            counts[i] += counts[i - 1];
        printf("%f ", counts[i]);
    }

    printf("\n");

    printf("MEAN RESIDUAL =  %f\n", sum / ssize);
}


void train_points_pca(Aligner *aligner, SampleSet *set){
    int ptsSize = set->ptsSize;
    int ssize = set->ssize;

    int WINW = set->WINW;
    int WINH = set->WINH;
    int sq = WINW * WINH;

    int featDim = aligner->oneDim * ptsSize;
    int regDim = 0;

    Shape meanShape = aligner->meanShape;

    float *rfeats = new float[ssize * featDim];
    for(int i = 0; i < ssize; i++){
        uint8_t *img = new uint8_t[sq];

        float *ptrFeats = rfeats + i * featDim;

        Sample *sample = set->samples[i];
        Shape oriShape;
        TranArgs arg;

        memcpy(img, sample->img, sizeof(uint8_t) * WINW * WINH);
        oriShape = sample->oriShape;

        similarity_transform(oriShape, meanShape, arg);
        arg.cen2.x = WINW >> 1;
        arg.cen2.y = WINH >> 1;

        affine_sample(img, WINW, WINH, WINW, oriShape, arg);

        extract_points_based_features(img, WINW, WINH, WINW, oriShape, aligner->gradDirect, ptrFeats);

        delete [] img;
        printf("%d\r", i), fflush(stdout);
    }

    aligner->pointsPCA = new PCAModel*[ptsSize];

    for(int i = 0; i < ptsSize; i++){
        aligner->pointsPCA[i] = train_pca(rfeats + i * aligner->oneDim, aligner->oneDim, ssize, featDim, 0.5f);
        regDim += aligner->pointsPCA[i]->dim;
    }

    aligner->featDim = regDim;

    printf("%d %d\n", featDim, regDim);

    delete [] rfeats;
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


PCAModel* train_feats_pca(Aligner *aligner, SampleSet *set){
    PCAModel *pcaModel;

    int ptsSize = set->ptsSize;
    int ssize = set->ssize;
    int WINW = set->WINW;
    int WINH = set->WINH;
    int sq = WINW * WINH;
    int featDim = aligner->featDim;

    Shape meanShape = aligner->meanShape;

    float *rfeats = new float[ssize * featDim];
    float *featBuffer = new float[aligner->oneDim * aligner->ptsSize];

    printf("EXTRACT ORIGIN SHAPE FEATURES\n");

    for(int i = 0; i < ssize; i++){
        uint8_t *img = new uint8_t[sq];

        float *ptrFeats = rfeats + i * featDim;

        Sample *sample = set->samples[i];
        Shape oriShape;
        TranArgs arg;

        memcpy(img, sample->img, sizeof(uint8_t) * WINW * WINH);
        oriShape = sample->oriShape;

        similarity_transform(oriShape, meanShape, arg);
        arg.cen2.x = WINW >> 1;
        arg.cen2.y = WINH >> 1;

        affine_sample(img, WINW, WINH, WINW, oriShape, arg);

        extract_points_based_features(img, WINW, WINH, WINW, oriShape, aligner->gradDirect, featBuffer);

        int ret = project_points_features(featBuffer, aligner->oneDim, aligner->pointsPCA, aligner->ptsSize, ptrFeats);
        assert(ret == featDim);

        delete [] img;
        printf("%d\r", i), fflush(stdout);
    }

    printf("TRAIN PCA\n");
    pcaModel = train_pca(rfeats, featDim, ssize, featDim, 0.95);

    delete [] rfeats;
    delete [] featBuffer;


    return pcaModel;
}


int save_features(const char *filePath, float *feats, float *rsdls, int ssize, int featDim, int ptsSize2){
    FILE *fout = fopen(filePath, "wb");

    if(fout == NULL){
        printf("Can't write file %s\n", filePath);
        return 1;
    }

    int ret;

    ret = fwrite(&ssize, sizeof(int), 1, fout);
    ret = fwrite(&featDim, sizeof(int), 1, fout);
    ret = fwrite(&ptsSize2, sizeof(int), 1, fout);

    ret = fwrite(feats, sizeof(float), ssize * featDim, fout);
    ret = fwrite(rsdls, sizeof(float), ssize * ptsSize2, fout);

    fclose(fout);

    return 0;
}


int load_features(const char *filePath, float **rfeats, float **rrsdls, int &ssize, int sfeatDim, int sptsSize2){
    FILE *fin = fopen(filePath, "rb");

    if(fin == NULL){
        printf("Can't read file %s\n", filePath);
        return 1;
    }

    int ret;
    float *feats, *rsdls;
    int featDim, ptsSize2;

    ret = fread(&ssize, sizeof(int), 1, fin);
    ret = fread(&featDim, sizeof(int), 1, fin);
    ret = fread(&ptsSize2, sizeof(int), 1, fin);

    assert(featDim == sfeatDim);
    assert(ptsSize2 == sptsSize2);

    feats = new float[ssize * featDim];
    rsdls = new float[ssize * ptsSize2];

    ret = fread(feats, sizeof(float), ssize * featDim, fin);  assert(ret == ssize * featDim);
    ret = fread(rsdls, sizeof(float), ssize * ptsSize2, fin); assert(ret == ssize * ptsSize2);

    fclose(fin);

    *rfeats = feats;
    *rrsdls = rsdls;

    return 0;
}


void initialize_aligner(Aligner **raligner, Shape &meanShape, int flag, int stage){
    char filePath[256], command[256];
    int ret;

    Aligner *aligner = NULL;

    for(int s = stage - 1; s >= 0; s--){
        sprintf(filePath, "cascade_%d.dat", s);

        if(load(filePath, &aligner) == 0)
            break;
    }

    if(aligner == NULL){
        aligner = new Aligner;
        memset(aligner, 0, sizeof(Aligner));

        aligner->gradDirect = GRAD_DIRECT;

        aligner->isAlign = flag;

        aligner->meanShape = meanShape;
        aligner->ptsSize = aligner->meanShape.ptsSize;
        aligner->oneDim = PYRAMID_SIZE * GRAD_DIRECT * CELL_NUM;
    }

    if(aligner->stage == stage){
        return;
    }
    else {
        float **coeffs = new float*[stage];
        PCAModel **rsdlPCA = new PCAModel*[stage];

        if(aligner->stage > 0){
            for(int i = 0; i < aligner->stage; i++){
                coeffs[i] = aligner->coeffs[i];
                rsdlPCA[i]= aligner->rsdlPCA[i];
            }

            delete [] aligner->rsdlPCA;
            delete [] aligner->coeffs;
        }

        aligner->coeffs = coeffs;
        aligner->rsdlPCA = rsdlPCA;
    }

    ret = system("mkdir -p model");
    ret = system("mkdir -p log");

    *raligner = aligner;
}


void train(const char *listFile, int flag, int stage, Aligner **resAligner){
    Aligner *aligner = NULL;
    SampleSet *set;

    int ret;
    char outPath[256];

    printf("READ SMAPLES\n");
    if(load(SAMPLE_FILE, &set) != 0){
        Shape meanShape;
        printf("LOAD MEAN SHAPE\n");
        ret = read_mean_shape(listFile, meanShape, SAMPLE_IMAGE_SIZE); assert(ret == 0);
        printf("LOAD SAMPLE IMAGES\n");
        ret = read_samples(listFile, meanShape, SAMPLE_IMAGE_SIZE * 2.3, SAMPLE_IMAGE_SIZE * 2.3, MIRROR, &set); assert(ret == 0);
    }

    initialize_aligner(&aligner, set->meanShape, flag, stage);
    init_grad_table(aligner->gradDirect);

    printf("GRAD DIRECT: %d, SAMPLE SIZE: %d, PTS SIZE: %d, FEATURE DIME: %d, TIMES: %d\n", aligner->gradDirect, set->ssize, aligner->ptsSize,
            aligner->featDim, TIMES);

    if(aligner->shapePCA == NULL){
        printf("TRAIN SHAPE PCA MODEL\n");
        aligner->shapePCA = train_shape_pca(set);
    }

    if(aligner->pointsPCA == NULL){
        printf("TRAIN POINT PCA MODEL\n");
        train_points_pca(aligner, set);
    }

    for(int s = aligner->stage; s < stage; s++){
        printf("---------------------- CASCADE %d ---------------------------\n", s);
        int ssize = set->ssize * TIMES;
        int ptsSize2 = aligner->ptsSize << 1;
        int featDim = aligner->featDim;

        PCAModel *rsdlPCA = NULL;

        float *feats = new float[ssize * featDim];
        float *rsdls = new float[ssize * ptsSize2];

        float rsdl[202];
        int id = 0;

        printf("GENERATE FEATURES\n");
        if(aligner->isAlign)
            generate_features_for_align(aligner, set, TIMES, feats, rsdls);
        else
            generate_features_for_track(aligner, set, TIMES, feats, rsdls);

        printf("CALCULATE MEAN RESIDUALS\n");
        mean_residual(rsdls, ssize, aligner->ptsSize);

        release(&set);

        printf("TRAIN RESIDUAL PCA\n");
        rsdlPCA = train_pca(rsdls, ptsSize2, ssize, ptsSize2, 0.8);

        printf("PROJECTION OF RESIDUALS\n");
        for(int i = 0; i < ssize; i++){
            pca_projection_feature(rsdls + i * ptsSize2, ptsSize2, rsdlPCA->dim, rsdlPCA->vmean, rsdlPCA->eigenVectors, rsdl);
            memcpy(rsdls + id, rsdl, sizeof(float) * rsdlPCA->dim);
            id += rsdlPCA->dim;
        }

        printf("SAVE FEATURES\n");
        ret = save_features(FEATURE_FILE, feats, rsdls, ssize, featDim, rsdlPCA->dim); assert(ret == 0);

        delete [] feats;
        delete [] rsdls;

        printf("SOLVE EQUATION\n");
        float *coeff = new float[featDim * rsdlPCA->dim]; assert(coeff != NULL);

        if(aligner->isAlign)
            solve_linear_equation_liblinear(FEATURE_FILE, coeff);
        else
            solve_linear_equation(FEATURE_FILE, coeff);


        aligner->coeffs[aligner->stage] = coeff;
        aligner->rsdlPCA[aligner->stage] = rsdlPCA;

        //save model
        aligner->stage++;

        sprintf(outPath, "model/cascade_%d.dat", s);

        save(outPath, aligner);

        assert(load(SAMPLE_FILE, &set) == 0);

        if(s == stage - 1){
            ssize = set->ssize * 1;

            feats = new float[ssize * featDim];
            rsdls = new float[ssize * ptsSize2];

            printf("GENERATE FEATURES\n");
            if(aligner->isAlign)
                generate_features_for_align(aligner, set, 1, feats, rsdls);
            else
                generate_features_for_track(aligner, set, 1, feats, rsdls);

            printf("CALCULATE MEAN RESIDUALS\n");
            mean_residual(rsdls, ssize, aligner->ptsSize);

            delete [] feats;
            delete [] rsdls;
        }

        printf("------------------------------------------------------------\n");
    }

    *resAligner = aligner;
    release(&set);
}


static void sdm_predict(float *coeff, int featDim, PCAModel *rsdlPCA, float *feat, float *shape){
    float delta[202];
    int dim = rsdlPCA->dim;
    int ptsSize2 = rsdlPCA->featDim;
    float *meanResidual = rsdlPCA->vmean;
    float *eigenMatrix = rsdlPCA->eigenVectors;


    memset(delta, 0, sizeof(float) * dim);
    for(int i = 0; i < featDim; i++){
        float t = feat[i];

        float *ptrShape = delta;
        int p = 0;

        for(; p <= dim - 8; p += 8){
            ptrShape[0] += t * coeff[0];
            ptrShape[1] += t * coeff[1];
            ptrShape[2] += t * coeff[2];
            ptrShape[3] += t * coeff[3];

            ptrShape[4] += t * coeff[4];
            ptrShape[5] += t * coeff[5];
            ptrShape[6] += t * coeff[6];
            ptrShape[7] += t * coeff[7];

            ptrShape += 8;
            coeff += 8;
        }

        for(; p < dim; p++){
            ptrShape[0] += t * coeff[0];
            ptrShape++;
            coeff ++;
        }
    }

    for(int i = 0; i < dim; i++){
        float t = delta[i];

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

    float *feat, *oriFeat;
    uint8_t *patch, *src;
    int winSize = SAMPLE_IMAGE_SIZE * 2 + 1;
    float scale;

    assert(aligner != NULL && img != NULL);

    feat = new float[featDim];
    oriFeat = new float[aligner->oneDim * aligner->ptsSize];


    srect = get_shape_rect(curShape);

    rect.x = srect.x - (srect.width  >> 1);
    rect.y = srect.y - (srect.height >> 1);
    rect.width  = srect.width * 2 + 1;
    rect.height = srect.height * 2 + 1;

    patch = new uint8_t[rect.width * rect.height + winSize * winSize];
    src = patch + rect.width * rect.height;

    extract_area_from_image(img, width, height, stride, patch, rect);

    scale = winSize / (float)rect.width;

    resizer_bilinear_gray(patch, rect.width, rect.height, rect.width, src, winSize, winSize, winSize);

    for(int p = 0; p < ptsSize; p++){
        curShape.pts[p] = (curShape.pts[p] - rect.x) * scale;
        curShape.pts[p + ptsSize] = (curShape.pts[p + ptsSize] - rect.y) * scale;
    }

    baseShape = curShape;

    for(int s = 0; s < aligner->stage; s++){
        PCAModel *rsdlPCA = aligner->rsdlPCA[s];

        memcpy(patch, src, sizeof(uint8_t) * winSize * winSize);

        similarity_transform(curShape, aligner->meanShape, arg);
        arg.cen2.x = winSize >> 1;
        arg.cen2.y = winSize >> 1;

        affine_sample(patch, winSize, winSize, winSize, curShape, arg);

        extract_points_based_features(patch, winSize, winSize, winSize, curShape, aligner->gradDirect, oriFeat);
        project_points_features(oriFeat, aligner->oneDim, aligner->pointsPCA, aligner->ptsSize, feat);

        sdm_predict(aligner->coeffs[s], featDim, rsdlPCA, feat, curShape.pts);

        if(!aligner->isAlign)
            denoise_shape(aligner->shapePCA, curShape, aligner->meanShape);


        arg.scale = 1.0f / arg.scale;
        arg.angle = -arg.angle;

        HU_SWAP(arg.cen1, arg.cen2, HPoint2f);
        affine_shape(curShape, curShape, arg);
    }

    //smooth_shape(baseShape, curShape);
    //show_shape(src, winSize, winSize, winSize, curShape, "cur");

    scale = 1.0f / scale;
    for(int p = 0; p < ptsSize; p++){
        curShape.pts[p] = curShape.pts[p] * scale + rect.x;
        curShape.pts[p + ptsSize] = curShape.pts[p + ptsSize] * scale + rect.y;
    }

    delete [] patch;
    delete [] oriFeat;
    delete [] feat;
}


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, HRect &srect, Shape &curShape){
    int ptsSize = aligner->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int featDim = aligner->featDim;
    TranArgs arg;
    HRect rect;
    int winSize = SAMPLE_IMAGE_SIZE * 2 + 1;
    float scale;

    float *feats = new float[featDim];
    float *oriFeats = new float[aligner->oneDim * aligner->ptsSize];

    assert(aligner != NULL && img != NULL);

    rect.x = srect.x - (srect.width  >> 1);
    rect.y = srect.y - (srect.height >> 1);
    rect.width  = srect.width << 1;
    rect.height = srect.height << 1;

    uint8_t *patch = new uint8_t[rect.width * rect.height + winSize * winSize];
    uint8_t *src = patch + rect.width * rect.height;

    extract_area_from_image(img, width, height, stride, patch, rect);

    scale = (float)rect.width / winSize;

    resizer_bilinear_gray(patch, rect.width, rect.height, rect.width, src, winSize, winSize, winSize);

    curShape = aligner->meanShape;

    for(int p = 0; p < ptsSize2; p++)
        curShape.pts[p] += SAMPLE_IMAGE_SIZE;

    for(int s = 0; s < aligner->stage; s++){
        PCAModel *rsdlPCA = aligner->rsdlPCA[s];

        memcpy(patch, src, sizeof(uint8_t) * winSize * winSize);

        similarity_transform(curShape, aligner->meanShape, arg);

        arg.cen2.x = winSize >> 1;
        arg.cen2.y = winSize >> 1;
        affine_sample(patch, winSize, winSize, winSize, curShape, arg);

        extract_points_based_features(patch, winSize, winSize, winSize, curShape, aligner->gradDirect, oriFeats);
        project_points_features(oriFeats, aligner->oneDim, aligner->pointsPCA, aligner->ptsSize, feats);

        sdm_predict(aligner->coeffs[s], featDim, rsdlPCA, feats, curShape.pts);

        arg.scale = 1.0f / arg.scale;
        arg.angle = -arg.angle;

        HU_SWAP(arg.cen1, arg.cen2, HPoint2f);
        affine_shape(curShape, curShape, arg);
    }

    for(int p = 0; p < ptsSize; p++){
        curShape.pts[p] = curShape.pts[p] * scale + rect.x;
        curShape.pts[p + ptsSize] = curShape.pts[p + ptsSize] * scale + rect.y;
    }

    delete [] patch;
    delete [] feats;
    delete [] oriFeats;
}

static int save_coeff(float *coeff, int rows, int cols, FILE *fout){
    if(fout == NULL || coeff == NULL || cols > rows)
        return 1;

    uint8_t *arr;
    int ret;

    ret = fwrite(&rows, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&cols, sizeof(int), 1, fout); assert(ret == 1);

    arr = (uint8_t*)malloc(sizeof(uint8_t) * cols);

    for(int y = 0; y < rows; y++){
        float minv = FLT_MAX, maxv = -FLT_MAX;
        float step = 0.0f;

        for(int x = 0; x < cols; x++){
            minv = HU_MIN(coeff[x], minv);
            maxv = HU_MAX(coeff[x], maxv);
        }

        step = (maxv - minv) / 255;

        ret = fwrite(&minv, sizeof(float), 1, fout); assert(ret == 1);
        ret = fwrite(&step, sizeof(float), 1, fout); assert(ret == 1);

        for(int x = 0; x < cols; x++)
            arr[x] = (coeff[x] - minv) / step;

        ret = fwrite(arr, sizeof(uint8_t), cols, fout); assert(ret == cols);

        coeff += cols;
    }

    free(arr);

    return 0;
}


int save(const char *filePath, Aligner *aligner){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;

    printf("SAVE MODEL %s\n", filePath);

    ret = fwrite(&aligner->stage, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&aligner->isAlign, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&aligner->ptsSize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&aligner->featDim, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&aligner->gradDirect, sizeof(int), 1, fout); assert(ret == 1);


    ret = fwrite(&aligner->meanShape, sizeof(Shape), 1, fout); assert(ret == 1);

    for(int i = 0; i < aligner->ptsSize; i++){
        ret = save_pca_model(fout, aligner->pointsPCA[i]); assert(ret == 0);
    }

    for(int i = 0; i < aligner->stage; i++){
        ret = save_pca_model(fout, aligner->rsdlPCA[i]); assert(ret == 0);
        ret = save_coeff(aligner->coeffs[i], aligner->featDim, aligner->rsdlPCA[i]->dim, fout); assert(ret == 0);
    }

    ret = save_pca_model(fout, aligner->shapePCA);

    assert(ret == 0);

    fclose(fout);

    return 0;
}


int load_coeff(float *coeff, int rows, int cols, FILE *fin){
    if(fin == NULL || coeff == NULL)
        return 1;

    int ret;
    uint8_t *arr = NULL;
    float *ptr = NULL;

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
            ptr[x] = arr[x] * step + minv;

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

    printf("LOAD MODEL %s\n", filePath);

    ret = fread(&aligner->stage, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->isAlign, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->ptsSize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->featDim, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&aligner->gradDirect, sizeof(int), 1, fin); assert(ret == 1);


    aligner->oneDim = aligner->gradDirect * CELL_NUM * PYRAMID_SIZE;

    ret = fread(&aligner->meanShape, sizeof(Shape), 1, fin); assert(ret == 1);

    aligner->pointsPCA = new PCAModel*[aligner->ptsSize];
    aligner->coeffs = new float*[aligner->stage];
    aligner->rsdlPCA = new PCAModel*[aligner->stage];

    for(int i = 0; i < aligner->ptsSize; i++){
        ret = load_pca_model(fin, &aligner->pointsPCA[i]); assert(ret == 0);
    }

    for(int i = 0; i < aligner->stage; i++){
        ret = load_pca_model(fin, &aligner->rsdlPCA[i]); assert(ret == 0);
        aligner->coeffs[i] = new float[aligner->featDim * aligner->rsdlPCA[i]->dim];
        ret = load_coeff(aligner->coeffs[i], aligner->featDim, aligner->rsdlPCA[i]->dim, fin); assert(ret == 0);
    }

    ret = load_pca_model(fin, &aligner->shapePCA); assert(ret == 0);

    fclose(fin);

    *raligner = aligner;

    init_grad_table(aligner->gradDirect);

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

        for(int i = 0; i < aligner->ptsSize; i++)
            release(&aligner->pointsPCA[i]);

        delete [] aligner->rsdlPCA;
        delete [] aligner->coeffs;
        delete [] aligner->pointsPCA;
    }

    aligner->coeffs = NULL;
    aligner->stage = 0;
}


void release(Aligner **aligner){
    if(*aligner == NULL)
        return ;

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
    if(fabs(dx) < 0.2)
        rate = 0;
    else if(fabs(dx) < 0.5)
        rate = 0.6;
    else if(fabs(dx) < 0.8)
        rate = 0.8;
    else if(fabs(dx) < 1.0)
        rate = 0.9;

    ptr = residual;
    for(int i = 0; i < ptsSize; i++){
        ptr[i] *= rate;
        float value = fabs(ptr[i]);
        if (value > 3){}
        else if(value > 2)
            ptr[i] *= 0.9;
        else if(value > 1)
            ptr[i] *= 0.8;
        else if(value > 0.5)
            ptr[i] *= 0.6;
        else if(value > 0.2)
            ptr[i] *= 0.4;
        else
            ptr[i] = 0;
    }

    rate = 1;
    if(fabs(dy) < 0.2)
        rate = 0;
    else if(fabs(dy) < 0.5)
        rate = 0.6;
    else if(fabs(dy) < 0.8)
        rate = 0.8;
    else if(fabs(dy) < 1.0)
        rate = 0.9;

    ptr = residual + ptsSize;
    for(int i = 0; i < ptsSize; i++){
        ptr[i] *= rate;
        float value = fabs(ptr[i]);
        if (value > 3){}
        else if(value > 2)
            ptr[i] *= 0.9;
        else if(value > 1)
            ptr[i] *= 0.8;
        else if(value > 0.5)
            ptr[i] *= 0.6;
        else if(value > 0.2)
            ptr[i] *= 0.4;
        else
            ptr[i] = 0;
    }


    for(int i = 0; i < ptsSize; i++){
        curShape[i] = lastShape[i] - residual[i];
        curShape[i + ptsSize] = lastShape[i + ptsSize] - residual[i + ptsSize];
    }
}
