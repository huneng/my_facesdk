#include "feature.h"

#define FEATURE_FILE "log/traindata.bin"


#define FEATURE_TIME 4
#define FEATURE_NUM 700

const int VALUE_LENGTH = 511;

static void gen_feature_type(Shape MEAN_SHAPE, int featNum, int ptsSize, float radius, int width, int height, FeatType* featTypes)
{
    float r = SAMPLE_IMAGE_SIZE * 0.5f * radius;

    cv::RNG rng(cv::getTickCount());

    float minx = (width >> 1) - 0.9f * SAMPLE_IMAGE_SIZE;
    float maxx = (width >> 1) + 0.9f * SAMPLE_IMAGE_SIZE;
    float miny = (height >> 1) - 0.9f * SAMPLE_IMAGE_SIZE;
    float maxy = (height >> 1) + 0.9f * SAMPLE_IMAGE_SIZE;

    float minDist = 0.05 * SAMPLE_IMAGE_SIZE;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < featNum; i++){
        FeatType *res = featTypes + i;

        float len, angle;
        float dist = 0;
        HPoint2f pta, ptac, ptb, ptbc;

        do {
            res->pntIdx1 = rng.uniform(0, ptsSize);

            pta.x = MEAN_SHAPE.pts[res->pntIdx1];
            pta.y = MEAN_SHAPE.pts[res->pntIdx1 + ptsSize];

            len = rng.uniform(0.0f, r);
            angle = rng.uniform(-HU_PI, HU_PI);

            res->off1X = len * cos(angle);
            res->off1Y = len * sin(angle);

            ptac.x = pta.x + res->off1X;
            ptac.y = pta.y + res->off1Y;

            if(ptac.x >= minx && ptac.x <= maxx && ptac.y >= miny && ptac.y <= maxy)
                break;
        } while(1);

        do{
            res->pntIdx2 = rng.uniform(0, ptsSize);

            ptb.x = MEAN_SHAPE.pts[res->pntIdx2];
            ptb.y = MEAN_SHAPE.pts[res->pntIdx2 + ptsSize];

            len = rng.uniform(0.0f, r);
            angle = rng.uniform(-HU_PI, HU_PI);

            res->off2X = len * cos(angle);
            res->off2Y = len * sin(angle);

            ptbc.x = ptb.x + res->off2X;
            ptbc.y = ptb.y + res->off2Y;

            dist = sqrt((ptac.x - ptbc.x) * (ptac.x - ptbc.x) + (ptac.y - ptbc.y) * (ptac.y - ptbc.y));

            if(ptbc.x >= minx && ptbc.x <= maxx && ptbc.y >= miny && ptbc.y <= maxy && dist > minDist)
                break;
        }while(1);

        if(res->pntIdx1 > res->pntIdx2){
            HU_SWAP(res->pntIdx1, res->pntIdx2, uint8_t);
            HU_SWAP(res->off1X, res->off2X, float);
            HU_SWAP(res->off1Y, res->off2Y, float);
        }
    }

    /* ---------- Test code ------------

    FILE *fout = fopen("log/FEATYPE.txt", "w");

    fprintf(fout, "radius: %f\n", r);
    for(int i = 0; i < featNum; i++){
        FeatType *ft = featTypes + i;
        fprintf(fout, "%3d %3d %10.6f %10.6f %10.6f %10.6f\n", ft->pntIdx1, ft->pntIdx2, ft->off1X, ft->off1Y, ft->off2X, ft->off2Y);
    }

    fclose(fout);

    cv::Mat img(height, width, CV_8UC3, cv::Scalar::all(255));


    for(int i = 0; i < featNum; i++){
        FeatType *ft = featTypes + i;

        float cx = MEAN_SHAPE.pts[ft->pntIdx1];
        float cy = MEAN_SHAPE.pts[ft->pntIdx1 + MEAN_SHAPE.ptsSize];

        float x0 = cx + ft->off1X;
        float y0 = cy + ft->off1Y;

        cv::circle(img, cv::Point2f(x0, y0), 1, cv::Scalar(0, 255, 0), -1);

        float x1 = cx + ft->off2X;
        float y1 = cy + ft->off2Y;

        cv::circle(img, cv::Point2f(x1, y1), 1, cv::Scalar(255, 0, 0), -1);
    }

    for(int i = 0; i < MEAN_SHAPE.ptsSize; i++)
        cv::circle(img, cv::Point2f(MEAN_SHAPE.pts[i], MEAN_SHAPE.pts[i + MEAN_SHAPE.ptsSize]), 2, cv::Scalar(0, 0, 255), -1);

    cv::rectangle(img, cv::Rect((width>> 1) - SAMPLE_IMAGE_SIZE, (height >> 1) - SAMPLE_IMAGE_SIZE, SAMPLE_IMAGE_SIZE * 2, SAMPLE_IMAGE_SIZE * 2), cv::Scalar(0, 0, 0), 2);
    cv::rectangle(img, cv::Rect(minx, miny, maxx - minx + 1, maxy - miny + 1), cv::Scalar(0, 0, 0), 2);
    cv::imwrite("log/featType.jpg", img);
    exit(0);
    // */
}


static int diff_feature(uint8_t *img, int stride, Shape &shape, FeatType &featType)
{
    int x, y;
    int a, b;

    HPoint2f pointA, pointB;

    pointA.x = shape.pts[featType.pntIdx1];
    pointA.y = shape.pts[featType.pntIdx1 + shape.ptsSize];

    pointB.x = shape.pts[featType.pntIdx2];
    pointB.y = shape.pts[featType.pntIdx2 + shape.ptsSize];

    //point 1
    x = pointA.x + featType.off1X;
    y = pointA.y + featType.off1Y;

    a = img[y * stride + x];

    //point 2
    x = pointB.x + featType.off2X;
    y = pointB.y + featType.off2Y;

    b = img[y * stride + x];

    return a - b;
}


static void binary_classify_loss2(float *posFeats, float *offsets, int size, float rate, float &bestThresh, double &minLoss){
    float maxFeatValue, minFeatValue;
    float featStep, rfeatStep;

    double v2[VALUE_LENGTH];
    double v[VALUE_LENGTH];

    int count[VALUE_LENGTH];

    maxFeatValue = -FLT_MAX, minFeatValue = FLT_MAX;

    for(int i = 0; i < size; i++){
        maxFeatValue = HU_MAX(maxFeatValue, posFeats[i]);
        minFeatValue = HU_MIN(minFeatValue, posFeats[i]);
    }

    featStep = (maxFeatValue - minFeatValue) / (VALUE_LENGTH - 1);
    assert(featStep > 0);

    memset(v2, 0, sizeof(double) * VALUE_LENGTH);
    memset(v, 0, sizeof(double) * VALUE_LENGTH);
    memset(count, 0, sizeof(int) * VALUE_LENGTH);

    rfeatStep = 1.0f / featStep;

    for(int i = 0; i < size; i++){
        int idx = (posFeats[i] -  minFeatValue) * rfeatStep;

        float off = offsets[i];

        v2[idx] += off * off;
        v[idx] += off;

        count[idx] ++ ;
    }

    for(int i = 1; i < VALUE_LENGTH; i++){
        v2[i] += v2[i - 1];
        v[i] += v[i - 1];

        count[i] += count[i - 1];
    }

    double cumv2 = v2[VALUE_LENGTH - 1];
    double cumv = v[VALUE_LENGTH - 1];
    double cumc = size;

    int CLASS_MIN_SIZE = cumc * rate;

    for(int i = 0; i < VALUE_LENGTH; i++){
        double lv2 = v2[i];
        double lv = v[i];
        int lc = count[i];

        double rv2 = cumv2 - lv2;
        double rv = cumv - lv;

        int rc = cumc - lc;

        double t, lvar, rvar, var;

        if(lc < CLASS_MIN_SIZE) continue;
        if(rc < CLASS_MIN_SIZE) break;

        t = lv / lc;
        lvar = lv2 / lc - t * t;

        t = rv / rc;
        rvar = rv2 / rc - t * t;

        var = HU_MAX(lvar, rvar);

        if(var < minLoss){
            minLoss = var;
            bestThresh = i * featStep + minFeatValue;
        }
    }
}


static void binary_classify_loss(float *posFeats, float *offsets, int size, float rate, float &bestThresh, double &minLoss){
    float minv = FLT_MAX, maxv = -FLT_MAX, meanv = 0;
    float step, rstep;

    int smaC1[VALUE_LENGTH], bigC1[VALUE_LENGTH];
    int smaC2[VALUE_LENGTH], bigC2[VALUE_LENGTH];
    for(int i = 0; i < size; i++){
        minv = HU_MIN(posFeats[i], minv);
        maxv = HU_MAX(posFeats[i], maxv);

        meanv += offsets[i];
    }

    if(maxv - minv < 5) return ;

    step = (maxv - minv) / (VALUE_LENGTH - 1);
    rstep = 1.0f / step;

    meanv /= size;

    memset(smaC1, 0, sizeof(int) * VALUE_LENGTH);
    memset(bigC1, 0, sizeof(int) * VALUE_LENGTH);

    for(int i = 0; i < size; i++){
        int idx = (posFeats[i] - minv) * rstep;

        smaC1[idx] += (offsets[i] <= meanv);
        bigC1[idx] += (offsets[i] > meanv);
    }

    memcpy(smaC2, smaC1, sizeof(int) * VALUE_LENGTH);
    memcpy(bigC2, bigC1, sizeof(int) * VALUE_LENGTH);

    int nonezero = (smaC1[0] + bigC1[0]) != 0;

    for(int i = 1; i < VALUE_LENGTH; i++){
        nonezero += (smaC1[i] + bigC1[i]) > 0;

        smaC1[i] += smaC1[i - 1];
        bigC1[i] += bigC1[i - 1];
    }

    if(nonezero < 5) return;

    for(int i = VALUE_LENGTH - 2; i >= 0; i --){
        smaC2[i] += smaC2[i + 1];
        bigC2[i] += bigC2[i + 1];
    }

    int CLASS_MIN_SIZE = rate * size;

    for(int i = 0; i < VALUE_LENGTH - 1; i++){
        int s1 = smaC1[i];
        int s2 = smaC2[i + 1];
        int b1 = bigC1[i];
        int b2 = bigC2[i + 1];

        if(s1 + b1 < CLASS_MIN_SIZE) continue;
        if(size - s1 - b1 < CLASS_MIN_SIZE) break;

        int a = s1 + b2;
        int b = s2 + b1;

        int c = HU_MIN(a, b);

        if(c < minLoss){
            minLoss = c;
            bestThresh = minv + i * step;
        }
    }
}


static void extract_features(SampleSet *set, int *idxs, int size, FeatType *featTypes, int fsize, float *feats){
    assert(set != NULL);
    assert(idxs != NULL);
    assert(feats != NULL);
    assert(featTypes != NULL);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < size; i++){
        Sample *sample = set->samples[idxs[i]];

        for(int j = 0, k = i; j < fsize; j++, k += size)
            feats[k] = diff_feature(sample->img, sample->stride, sample->curShape, featTypes[j]);
    }
}


static void init_rates(float *rate, int depth){
    assert(depth > 3 && depth < 7);

    if(depth == 4){
        rate[0] = 0.2;

        rate[1] = 0.1;
        rate[2] = 0.1;

        for(int i = 3; i < 7; i++)
            rate[i] = 0.05;
    }
    else if(depth == 5){
        rate[0] = 0.4;

        rate[1] = 0.3;
        rate[2] = 0.3;

        for(int i = 3; i < 7; i++)
            rate[i] = 0.15;

        for(int i = 7; i < 15; i++)
            rate[i] = 0.05;
    }
    else if(depth == 6){
        rate[0] = 0.4;

        rate[1] = 0.3;
        rate[2] = 0.3;

        for(int i = 3; i < 7; i++)
            rate[i] = 0.2;

        for(int i = 7; i < 15; i++)
            rate[i] = 0.1;

        for(int i = 15; i < 31; i ++)
            rate[i] = 0.05;
    }
}


typedef struct {
    double bError;
    float bThresh;
    int bIdx;
} ClassPair;



static void class_residual(float *posFeats, int posSize, float *residuals,
                 int featDim, float rate, ClassPair *pair){

    pair->bError = DBL_MAX;
    pair->bThresh = 0.0f;
    pair->bIdx = -1;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < featDim; i++){
        double error = DBL_MAX;
        float thresh = 0.0f;

        binary_classify_loss(posFeats + i * posSize, residuals, posSize, rate, thresh, error);

#pragma omp critical
        {
            if(error < pair->bError){
                pair->bError = error;
                pair->bThresh = thresh;
                pair->bIdx = i;
            }
        }
    }
}


typedef struct{
    int *posIdxs;
    int posSize;

    float *residuals;

} JNodePair;


static void split(float *posFeats, int16_t thresh, JNodePair *ppair, JNodePair *lpair, JNodePair *rpair){
    lpair->posSize = 0;
    rpair->posSize = 0;

    for(int i = 0; i < ppair->posSize; i++){
        if(posFeats[i] <= thresh){
            lpair->posIdxs[lpair->posSize] = ppair->posIdxs[i];
            lpair->residuals[lpair->posSize] = ppair->residuals[i];

            lpair->posSize++;
        }
        else{
            rpair->posIdxs[rpair->posSize] = ppair->posIdxs[i];
            rpair->residuals[rpair->posSize] = ppair->residuals[i];

            rpair->posSize++;
        }
    }

    int minLeafSize = 10;

    assert(lpair->posSize > minLeafSize );
    assert(rpair->posSize > minLeafSize );
}


void train_tree(JTree *root, int depth, SampleSet *posSet, float radius, float *rsdls){
    cv::RNG rng(cv::getTickCount());

    int posSize = 20000;

    int nlSize   = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);

    int nodeSize = nlSize - leafSize;

    float rates[32];

    float *posFeats = new float[posSize * FEATURE_NUM];
    float *bestPosFeats = new float[posSize];

    float *residuals = NULL;
    int *pIdxs = NULL, *nIdxs = NULL;

    FeatType *featTypes = new FeatType[FEATURE_NUM * FEATURE_TIME];

    JNodePair *pairs = new JNodePair[nlSize];

    memset(pairs, 0, sizeof(JNodePair) * nlSize);

    init_rates(rates, depth);

    for(int i = 0; i < nlSize; i++){
        pairs[i].posIdxs = new int[posSize];
        pairs[i].residuals = new float[posSize];
    }

    //root
    pIdxs = pairs[0].posIdxs;

    for(int i = 0; i < posSize; i++)
        pIdxs[i] = i;

    pairs[0].posSize = posSize;

    memcpy(pairs[0].residuals, rsdls, sizeof(float) * posSize);

    for(int i = 0; i < nodeSize; i++){
        JNode *node = root + i;
        int idL = i * 2 + 1;
        int idR = i * 2 + 2;

        float rate = rates[i];

        double bError;

        pIdxs = pairs[i].posIdxs;
        posSize = pairs[i].posSize;
        residuals = pairs[i].residuals;
        node->posSize = pairs[i].posSize;

        gen_feature_type(posSet->meanShape, FEATURE_NUM * FEATURE_TIME, posSet->ptsSize, radius, posSet->WINW, posSet->WINH, featTypes);

        bError = FLT_MAX;
        for(int k = 0; k < FEATURE_TIME; k++){
            ClassPair cpair;

            extract_features(posSet, pIdxs, posSize, featTypes + k * FEATURE_NUM, FEATURE_NUM, posFeats);

            class_residual(posFeats, posSize, residuals, FEATURE_NUM, rate, &cpair); assert(cpair.bIdx != -1);

            if(cpair.bError < bError){
                bError = cpair.bError;

                node->thresh = cpair.bThresh;
                node->featType = featTypes[cpair.bIdx + k * FEATURE_NUM];
                memcpy(bestPosFeats, posFeats + cpair.bIdx * posSize, sizeof(float) * posSize);
            }
        }

        split(bestPosFeats, node->thresh, pairs + i, pairs + idL, pairs + idR);

        node->left  = root + idL;
        node->right = root + idR;
    }

    delete [] bestPosFeats;
    delete [] posFeats;
    delete [] featTypes;

    for(int i = nodeSize; i < nlSize; i++){
        JNode *leaf = root + i;

        leaf->leafID = i - nodeSize;
        leaf->posSize = pairs[i].posSize;

        leaf->left  = NULL;
        leaf->right = NULL;
    }

    for(int i = 0; i < nlSize; i++){
        delete [] pairs[i].posIdxs;
        delete [] pairs[i].residuals;
    }

    delete [] pairs;
}


uint8_t predict(JTree *root, uint8_t *img, int stride, Shape &shape){
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    while(root->left != NULL)
        root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    return root->leafID;
}


void save(FILE *fout, int depth, JTree *root){
    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;
    int ret;

    assert(root != NULL && fout != NULL);

    for(int i = 0; i < nodeSize; i++){
        JNode *node = root + i;

        FeatType ft = node->featType;

        int16_t value;

        ret = fwrite(&ft.pntIdx1, sizeof(uint8_t), 1, fout); assert(ret == 1);
        ret = fwrite(&ft.pntIdx2, sizeof(uint8_t), 1, fout); assert(ret == 1);

        value = int16_t(ft.off1X * 100);
        ret = fwrite(&value, sizeof(int16_t), 1, fout); assert(ret == 1);
        value = int16_t(ft.off1Y * 100);
        ret = fwrite(&value, sizeof(int16_t), 1, fout); assert(ret == 1);

        value = int16_t(ft.off2X * 100);
        ret = fwrite(&value, sizeof(int16_t), 1, fout); assert(ret == 1);
        value = int16_t(ft.off2Y * 100);
        ret = fwrite(&value, sizeof(int16_t), 1, fout); assert(ret == 1);

        ret = fwrite(&node->thresh, sizeof(int16_t), 1, fout);
        assert(ret == 1);
    }
}


void load(FILE *fin, int depth, JTree *root){
    assert(fin != NULL && root != NULL);

    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;
    int ret;

    for(int i = 0; i < nodeSize; i++){
        JNode *node = root + i;
        FeatType ft;

        int16_t value;

        ret = fread(&ft.pntIdx1, sizeof(uint8_t), 1, fin); assert(ret == 1);
        ret = fread(&ft.pntIdx2, sizeof(uint8_t), 1, fin); assert(ret == 1);

        ret = fread(&value, sizeof(int16_t), 1, fin); assert(ret == 1);
        ft.off1X = value * 0.01f;

        ret = fread(&value, sizeof(int16_t), 1, fin); assert(ret == 1);
        ft.off1Y = value * 0.01f;

        ret = fread(&value, sizeof(int16_t), 1, fin); assert(ret == 1);
        ft.off2X = value * 0.01f;

        ret = fread(&value, sizeof(int16_t), 1, fin); assert(ret == 1);
        ft.off2Y = value * 0.01f;

        ret = fread(&node->thresh, sizeof(int16_t), 1, fin);
        assert(ret == 1);

        node->featType = ft;
        node->left = root + i * 2 + 1;
        node->right = root + i * 2 + 2;
    }

    JNode *leafs = root + nodeSize;

    for(int i = 0; i < leafSize; i++){
        leafs[i].leafID = i;
        leafs[i].left = NULL;
        leafs[i].right = NULL;
    }
}


void print_tree(FILE *fout, JTree *root, int depth){
    static int TREE_ID = 0;

    int nlSize   = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);

    int nodeSize = nlSize - leafSize;

    fprintf(fout, "TREE: %d\n", TREE_ID++);
    for(int i = 0; i < nodeSize; i++){
        JNode *node = root + i;

        fprintf(fout, "JNode: %2d, thresh: %3d, sample size: %d\n", i, node->thresh, node->posSize);
    }

    for(int i = 0; i < leafSize; i++){
        JNode *leaf = root + i + nodeSize;

        fprintf(fout, "Leaf: %2d, sample size: %d\n", i, leaf->posSize);
    }
}


void release(JTree **root){
    if(*root != NULL){
        delete [] *root;
    }

    *root = NULL;
}


#define SOLVE_TYPE double

static int solve_linear_equation_liblinear(const char *filePath, float *XMat){
    int ssize, leafSize, treeSize, ptsSize2, ret;
    int featDim;
    uint8_t *leafIDs;
    float *buffer;

    FILE *fin = fopen(filePath, "rb");

    assert(fin != NULL);

    ret = fread(&ssize, sizeof(int), 1, fin);
    ret = fread(&treeSize, sizeof(int), 1, fin);
    ret = fread(&leafSize, sizeof(int), 1, fin);
    ret = fread(&ptsSize2, sizeof(int), 1, fin);

    featDim = leafSize * treeSize;
    leafIDs = (uint8_t*)malloc(sizeof(uint8_t) * treeSize);
    buffer = (float*)malloc(sizeof(float) * ptsSize2);

    struct problem   *prob   = new struct problem[ptsSize2];
    struct parameter *params = new struct parameter[ptsSize2];

    struct feature_node **nodes = new struct feature_node*[ssize];
    double **R = new double*[ptsSize2];

    struct model **models = new struct model*[ptsSize2];


    for(int i = 0; i < ssize; i++){
        int offset = 1;
        nodes[i] = new struct feature_node[treeSize + 1];

        ret = fread(leafIDs, sizeof(uint8_t), treeSize, fin); assert(ret == treeSize);

        for(int j = 0; j < treeSize; j++){
            nodes[i][j].index = leafIDs[j] + offset;
            nodes[i][j].value = 1;

            offset += leafSize;
        }

        nodes[i][treeSize].index = -1;
        nodes[i][treeSize].value = -1;
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
        params[i].eps = 0.00001;
    }

    for(int j = 0; j < ssize; j++){
        ret = fread(buffer, sizeof(float), ptsSize2, fin); assert(ret == ptsSize2);

        for(int i = 0; i < ptsSize2; i++)
            R[i][j] =  buffer[i];
    }

    delete [] buffer;
    delete [] leafIDs;

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

    //write_matrix(XMat, ptsSize2, featDim, 1, "log/coeff.txt");

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


void extract_binary_feature(Forest *forest, uint8_t *img, int stride, Shape &shape, uint8_t *leafIDs){
    for(int i = 0; i < forest->treeSize; i++){
        leafIDs[i] = predict(forest->trees[i], img, stride, shape);
    }
}


void extract_binary_feature(Forest *forest, uint8_t *img, int stride, Shape &shape, float *binFeats){
    uint8_t leafID;

    memset(binFeats, 0, sizeof(float) * forest->featDim);

    for(int i = 0; i < forest->treeSize; i++){
        leafID = predict(forest->trees[i], img, stride, shape);

        binFeats[leafID] = 1;
        binFeats += forest->leafSize;
    }
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


void train_forest(SampleSet *set, int treeSize, int depth, float radius, Forest *forest){
    assert(set != NULL);
    assert(forest != NULL);

    assert(forest->trees == NULL);
    assert(forest->vmean == NULL);
    assert(forest->coeff == NULL);
    assert(forest->offsets == NULL);

    int ssize = set->ssize;
    int ptsSize = set->ptsSize;
    int ptsSize2 = set->ptsSize << 1;

    int nlSize = (1 << depth) - 1;

    PCAModel *rsdlPCA;

    float *rsdls, **ptrRsdls;
    uint8_t *leafIDs;

    forest->depth = depth;
    forest->treeSize = treeSize;
    forest->ptsSize = ptsSize;
    forest->leafSize = 1 << (forest->depth - 1);
    forest->featDim = treeSize * forest->leafSize;

    forest->trees = new JTree*[forest->treeSize];

    leafIDs = new uint8_t[forest->treeSize];
    rsdls = new float[ptsSize2 * set->ssize];
    ptrRsdls = new float*[set->ssize];

    printf("CALCULATE RESIDUALS\n");
    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];
        float *ptrRsdl = rsdls + i * ptsSize2;

        for(int p = 0; p < ptsSize2; p++)
            ptrRsdl[p] = sample->oriShape.pts[p] - sample->curShape.pts[p];

        ptrRsdls[i] = ptrRsdl;
    }

    printf("MEAN RESIDUALS\n");
    mean_residual(rsdls, set->ssize, ptsSize);

    printf("TRAIN RESIDUAL PCA\n");
    rsdlPCA = train_pca(rsdls, ptsSize2, set->ssize, ptsSize2, 0.7);

    forest->dim = rsdlPCA->dim;
    forest->vmean = new float[rsdlPCA->featDim];
    forest->coeff = new float[rsdlPCA->featDim * rsdlPCA->dim];

    memcpy(forest->vmean, rsdlPCA->vmean, sizeof(float) * rsdlPCA->featDim);
    memcpy(forest->coeff, rsdlPCA->eigenVectors, sizeof(float) * rsdlPCA->featDim * rsdlPCA->dim);

#if 1
    printf("TRAIN FORESTS\n");
    float *buffer = new float[set->ssize];

    for(int i = 0; i < forest->treeSize; i++){
        int pntIdx = i % ptsSize2;

        forest->trees[i] = new JNode[nlSize];

        random_samples(set, ptrRsdls);

        for(int s = 0; s < set->ssize; s++)
            buffer[s] = ptrRsdls[s][pntIdx];

        train_tree(forest->trees[i], forest->depth, set, radius, buffer);

        printf("%d\r", i); fflush(stdout);
    }

    delete [] buffer;
    release(&rsdlPCA);

    printf("SAVE DATA\n");
    FILE *fout = fopen(FEATURE_FILE, "wb");
    int ret;

    assert(fout != NULL);

    ret = fwrite(&set->ssize, sizeof(int), 1, fout);
    ret = fwrite(&forest->treeSize, sizeof(int), 1, fout);
    ret = fwrite(&forest->leafSize, sizeof(int), 1, fout);
    ret = fwrite(&forest->dim, sizeof(int), 1, fout);

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        extract_binary_feature(forest, sample->img, sample->stride, sample->curShape, leafIDs);

        ret = fwrite(leafIDs, sizeof(uint8_t), forest->treeSize, fout); assert(ret == forest->treeSize);
    }

    for(int i = 0; i < set->ssize; i++){
        pca_projection_feature(ptrRsdls[i], ptsSize2, forest->dim, forest->vmean, forest->coeff, ptrRsdls[i]);
        ret = fwrite(ptrRsdls[i], sizeof(float), forest->dim, fout); assert(ret == forest->dim);
    }

#else
    for(int i = 0; i < set->ssize; i++)
        pca_projection_feature(ptrRsdls[i], ptsSize2, forest->dim, forest->vmean, forest->coeff, ptrRsdls[i]);

    printf("TRAIN FORESTS\n");
    float *buffer = new float[set->ssize];
    int pntIdx = 0;
    float cum1 = 0, cum2 = 0;
    for(int i = 0; i < rsdlPCA->dim; i++)
        cum2 += rsdlPCA->eigenValues[i];

    cum1 = rsdlPCA->eigenValues[0];
    for(int i = 0; i < forest->treeSize; i++){
        if(i / float(forest->treeSize) > cum1 / cum2){
            pntIdx ++;
            cum1 += rsdlPCA->eigenValues[pntIdx];
        }

        forest->trees[i] = new JNode[nlSize];

        random_samples(set, ptrRsdls);

        for(int s = 0; s < set->ssize; s++)
            buffer[s] = ptrRsdls[s][pntIdx];

        train_tree(forest->trees[i], forest->depth, set, radius, buffer);

        printf("%3d %d\r", pntIdx, i); fflush(stdout);
    }

    delete [] buffer;
    release(&rsdlPCA);

    printf("SAVE DATA\n");
    FILE *fout = fopen(FEATURE_FILE, "wb");
    int ret;

    assert(fout != NULL);

    ret = fwrite(&set->ssize, sizeof(int), 1, fout);
    ret = fwrite(&forest->treeSize, sizeof(int), 1, fout);
    ret = fwrite(&forest->leafSize, sizeof(int), 1, fout);
    ret = fwrite(&forest->dim, sizeof(int), 1, fout);

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        extract_binary_feature(forest, sample->img, sample->stride, sample->curShape, leafIDs);

        ret = fwrite(leafIDs, sizeof(uint8_t), forest->treeSize, fout); assert(ret == forest->treeSize);
    }

    for(int i = 0; i < set->ssize; i++){
        ret = fwrite(ptrRsdls[i], sizeof(float), forest->dim, fout); assert(ret == forest->dim);
    }
#endif
    fclose(fout);

    delete [] rsdls;
    delete [] ptrRsdls;
    delete [] leafIDs;

    release_data(set);

    printf("SOLVE LINEAR EQUATIONS\n");
    forest->offsets = new float[forest->featDim * forest->dim];
    solve_linear_equation_liblinear(FEATURE_FILE, forest->offsets);
}


void predict(Forest *forest, uint8_t *img, int width, int height, int stride, Shape &curShape){
    float prsdl[MAX_PTS_SIZE2];
    float rsdl[MAX_PTS_SIZE2];
    float *coeff = forest->coeff;

    int ptsSize2 = forest->ptsSize << 1;

    memset(prsdl, 0, sizeof(float) * MAX_PTS_SIZE2);

    for(int i = 0, id = 0; i < forest->treeSize; i++, id += forest->leafSize){
        uint8_t leafID;
        float *delta;

        leafID = predict(forest->trees[i], img, stride, curShape);

        delta = forest->offsets + (id + leafID) * forest->dim;

        for(int d = 0; d < forest->dim; d ++)
            prsdl[d] += delta[d];
    }

    memcpy(rsdl, forest->vmean, sizeof(float) * ptsSize2);

    for(int d = 0; d < forest->dim; d++){
        float value = prsdl[d];

        for(int p = 0; p < ptsSize2; p++)
            rsdl[p] += value * coeff[p];

        coeff += ptsSize2;
    }

    for(int p = 0; p < ptsSize2; p++)
        curShape.pts[p] += rsdl[p];
}


static void write_offsets(FILE *fout, float *offsets, int dim, int binSize){
    assert(fout != NULL);
    assert(offsets != NULL);

    uint8_t buffer[MAX_PTS_SIZE2];
    int ret;

    for(int i = 0; i < binSize; i++){
        float minv = FLT_MAX, maxv = -FLT_MAX, step;

        for(int p = 0; p < dim; p++){
            minv = HU_MIN(minv, offsets[p]);
            maxv = HU_MAX(maxv, offsets[p]);
        }

        step = (maxv - minv) / 255;

        for(int p = 0; p < dim; p++)
            buffer[p] = uint8_t((offsets[p] - minv) / step);

        ret = fwrite(&minv, sizeof(float), 1, fout); assert(ret == 1);
        ret = fwrite(&step, sizeof(float), 1, fout); assert(ret == 1);
        ret = fwrite(buffer, sizeof(uint8_t), dim, fout); assert(ret == dim);

        offsets += dim;
    }

    //write_matrix(offsets - dim * binSize, dim, binSize, 10, "log/write_offsets.txt");
}


void save( FILE *fout, Forest *forest){
    assert(fout != NULL);
    assert(forest != NULL);

    int ret;

    ret = fwrite(&forest->treeSize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&forest->depth, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&forest->ptsSize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&forest->dim, sizeof(int), 1, fout); assert(ret == 1);

    write_offsets(fout, forest->offsets, forest->dim, forest->featDim);

    ret = fwrite(forest->vmean, sizeof(float), forest->ptsSize * 2, fout); assert(ret == forest->ptsSize * 2);
    ret = fwrite(forest->coeff, sizeof(float), forest->ptsSize * 2 * forest->dim, fout); assert(ret == forest->dim * forest->ptsSize * 2);

    for(int i = 0; i < forest->treeSize; i++)
        save(fout, forest->depth, forest->trees[i]);
}


static void read_offsets(FILE *fin, float *offsets, int dim, int binSize){
    assert(fin != NULL);
    assert(offsets != NULL);

    uint8_t buffer[MAX_PTS_SIZE2];
    int ret;

    for(int i = 0; i < binSize; i++){
        float minv, step;

        ret = fread(&minv, sizeof(float), 1, fin); assert(ret == 1);
        ret = fread(&step, sizeof(float), 1, fin); assert(ret == 1);
        ret = fread(buffer, sizeof(uint8_t), dim, fin); assert(ret == dim);

        for(int p = 0; p < dim; p++)
            offsets[p] = buffer[p] * step + minv;

        offsets += dim;
    }

    //write_matrix(offsets - dim * binSize, dim, binSize, 10, "log/read_offsets.txt");
}


void load(FILE *fin, Forest **forest){
    assert(fin != NULL);

    int ret, nlSize;

    Forest *f = new Forest;

    memset(f, 0, sizeof(Forest));

    ret = fread(&f->treeSize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&f->depth, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&f->ptsSize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&f->dim, sizeof(int), 1, fin); assert(ret == 1);

    f->leafSize = 1 << (f->depth - 1);
    f->featDim = f->leafSize * f->treeSize;

    f->offsets = new float[f->featDim * f->dim];
    f->vmean = new float[f->ptsSize * 2];
    f->coeff = new float[f->ptsSize * 2 * f->dim];

    read_offsets(fin, f->offsets, f->dim, f->featDim);

    ret = fread(f->vmean, sizeof(float), f->ptsSize * 2, fin); assert(ret == f->ptsSize * 2);
    ret = fread(f->coeff, sizeof(float), f->ptsSize * 2 * f->dim, fin); assert(ret == f->dim * f->ptsSize * 2);

    nlSize = (1 << f->depth) - 1;

    f->trees = new JTree*[f->treeSize];
    for(int i = 0; i < f->treeSize; i++){
        f->trees[i] = new JTree[nlSize];

        memset(f->trees[i], 0, sizeof(JTree) * nlSize);
        load(fin, f->depth, f->trees[i]);
    }

    *forest = f;
}


void release_data(Forest *forest){
    if(forest == NULL) return;

    for(int i = 0; i < forest->treeSize; i++)
        release(forest->trees + i);

    delete [] forest->trees;

    if(forest->offsets != NULL)
        delete [] forest->offsets;
    forest->offsets = NULL;

    if(forest->vmean != NULL)
        delete [] forest->vmean;
    forest->vmean = NULL;

    if(forest->coeff != NULL)
        delete [] forest->coeff;
    forest->coeff = NULL;

}


void release(Forest **forest){
    if(*forest == NULL) return;

    delete *forest;
    *forest = NULL;
}

