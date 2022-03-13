#include "tree.h"

#define FEATURE_TIME 10
#define FEATURE_NUM 200

#define USE_GINI
//#define USE_ENTROPY

const int VALUE_LENGTH = 511;


static void gen_feature_type(Shape MEAN_SHAPE, int featNum, int ptsSize, float radius, int width, int height, FeatType* featTypes)
{
    float r = HU_MAX(width, height) * 0.5f * radius;

    cv::RNG rng(cv::getTickCount());

    float minx = 0.1f * width;
    float maxx = 0.9f * width;
    float miny = 0.1f * height;
    float maxy = 0.9f * height;

    float minDist = 0.05 * width;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < featNum; i++){
        FeatType *res = featTypes + i;

        float len, angle;
        float dist = 0;
        HPoint2f pta, ptac, ptb, ptbc;

        do {
            res->pntIdx1 = rng.uniform(0, ptsSize);
            pta = MEAN_SHAPE.pts[res->pntIdx1];

            len = rng.uniform(0.0f, r);
            angle = rng.uniform(-HU_PI, HU_PI);

            res->off1X = len * cosf(angle);
            res->off1Y = len * sinf(angle);

            ptac.x = pta.x + res->off1X;
            ptac.y = pta.y + res->off1Y;

            if(ptac.x >= minx && ptac.x <= maxx && ptac.y >= miny && ptac.y <= maxy)
                break;
        } while(1);

        do{
            res->pntIdx2 = rng.uniform(0, ptsSize);
            ptb = MEAN_SHAPE.pts[res->pntIdx2];

            len = rng.uniform(0.0f, r);
            angle = rng.uniform(-HU_PI, HU_PI);

            res->off2X = len * cosf(angle);
            res->off2Y = len * sinf(angle);

            ptbc.x = ptb.x + res->off2X;
            ptbc.y = ptb.y + res->off2Y;

            dist = sqrtf((ptac.x - ptbc.x) * (ptac.x - ptbc.x) + (ptac.y - ptbc.y) * (ptac.y - ptbc.y));

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

        float cx = MEAN_SHAPE.pts[ft->pntIdx1].x;
        float cy = MEAN_SHAPE.pts[ft->pntIdx1].y;

        float x0 = cx + ft->off1X;
        float y0 = cy + ft->off1Y;

        cv::circle(img, cv::Point2f(x0, y0), 1, cv::Scalar(0, 255, 0), -1);

        float x1 = cx + ft->off2X;
        float y1 = cy + ft->off2Y;

        cv::circle(img, cv::Point2f(x1, y1), 1, cv::Scalar(255, 0, 0), -1);
    }

    for(int i = 0; i < MEAN_SHAPE.ptsSize; i++)
        cv::circle(img, cv::Point2f(MEAN_SHAPE.pts[i].x, MEAN_SHAPE.pts[i].y), 2, cv::Scalar(0, 0, 255), -1);

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

    pointA = shape.pts[featType.pntIdx1];
    pointB = shape.pts[featType.pntIdx2];

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


static int diff_feature(uint8_t *img, int stride, Shape &shape,
        FeatType &featType, TranArgs &arg)
{
    int a, b;
    int x, y;

    HPoint2f point;

    //point 1
    point = shape.pts[featType.pntIdx1];

    x = featType.off1X * arg.scosa + featType.off1Y * arg.ssina + point.x;
    y = featType.off1Y * arg.scosa - featType.off1X * arg.ssina + point.y;

    a = img[y * stride + x];

    //point 2
    point = shape.pts[featType.pntIdx2];

    x = featType.off2X * arg.scosa + featType.off2Y * arg.ssina + point.x;
    y = featType.off2Y * arg.scosa - featType.off2X * arg.ssina + point.y;

    b = img[y * stride + x];

    return a - b;
}


static int statistic(float *feats, double *weights, int size, float minValue, float rstep,  double *weightTable, int *countTable){
    double sumw = 0.0;
    int nonezero = 0;

    memset(weightTable, 0, sizeof(double) * VALUE_LENGTH);
    memset(countTable, 0, sizeof(int) * VALUE_LENGTH);

    for(int i = 0; i < size; i++){
        int id = (feats[i] - minValue) * rstep;

        weightTable[id] += weights[i];
        countTable[id] ++;

        sumw += weights[i];
    }

    sumw = 1.0 / sumw;

    weightTable[0] *= sumw;
    for(int i = 1; i < VALUE_LENGTH; i++){
        weightTable[i] *= sumw;
        weightTable[i] += weightTable[i - 1];

        nonezero += (countTable[i] != 0);
        countTable[i] += countTable[i - 1];
    }

    return nonezero;
}


static void binary_classify_error(float *posFeats, double *posWeights, int posSize,
        float *negFeats, double *negWeights, int negSize, float rate, float &thresh, double &error){
    double posWT[VALUE_LENGTH], negWT[VALUE_LENGTH];

    int pcount[VALUE_LENGTH], ncount[VALUE_LENGTH];

    float maxValue, minValue;
    float featStep, rfeatStep;

    int CLASS_MIN_SIZE_P = rate * posSize;
    int CLASS_MIN_SIZE_N = rate * negSize;

    int i;
    int len;

    maxValue = -FLT_MAX, minValue = FLT_MAX;

    for(i = 0; i < posSize; i++){
        maxValue = HU_MAX(maxValue, posFeats[i]);
        minValue = HU_MIN(minValue, posFeats[i]);
    }

    for(i = 0; i < negSize; i++){
        maxValue = HU_MAX(maxValue, negFeats[i]);
        minValue = HU_MIN(minValue, negFeats[i]);
    }

    maxValue = HU_MAX(minValue + 1, maxValue);

    featStep = (maxValue - minValue) / (VALUE_LENGTH - 1);
    assert(featStep > 0);

    rfeatStep = 1.0f / featStep;

    if(statistic(posFeats, posWeights, posSize, minValue, rfeatStep, posWT, pcount) < 5) return;
    if(statistic(negFeats, negWeights, negSize, minValue, rfeatStep, negWT, ncount) < 5) return;

#ifdef USE_GINI
    double sumW = 2.0;
#endif

    i = 0;
    len = VALUE_LENGTH - 1;
    for(; i < VALUE_LENGTH; i++)
        if(pcount[i + 1] >= CLASS_MIN_SIZE_P && ncount[i + 1] >= CLASS_MIN_SIZE_N)
            break;

    for(; len >= i; len --)
        if(posSize - pcount[len] >= CLASS_MIN_SIZE_P &&
                negSize - ncount[len] >= CLASS_MIN_SIZE_N)
            break;

    for(; i <= len; i++){
        double e = 0;
#if defined(USE_ENTROPY)
        double WL = (posWT[i] + 1.0 - negWT[i]) * 0.5;
        double WR = 1.0 - WL;

        e = - WL * log(WL) - WR * log(WR);

#elif defined(USE_GINI)
        double PWL = posWT[i];
        double NWL = negWT[i];

        double PWR = 1.0 - PWL;
        double NWR = 1.0 - NWL;

        double sumL = PWL + NWL;
        double sumR = PWR + NWR;

        e = ( (sumL / sumW) * (PWL / sumL) * (1.0 - PWL / sumL) +
                (sumR / sumW) * (PWR / sumR) * (1.0 - PWR / sumR) );

#else
        e = posWT[i] - (1.0 - negWT[i]);
        e = HU_MIN(e, 2.0 - e);
#endif

        if(e < error){
            error = e;
            thresh = minValue + i * featStep;
        }
    }
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

    int WINW = set->WINW;
    int WINH = set->WINH;

    assert(feats != NULL);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < size; i++){
        Sample *sample = set->samples[idxs[i]];

        for(int j = 0, k = i; j < fsize; j++, k += size)
            feats[k] = diff_feature(sample->img, sample->stride, sample->curShape, featTypes[j], sample->arg);
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


static void class_error(float *posFeats, double *posWeights, int posSize,
        float *negFeats, double *negWeights, int negSize,
        int featDim, float rate, ClassPair *pair){

    pair->bError = DBL_MAX;
    pair->bThresh = 0.0f;
    pair->bIdx = -1;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < featDim; i++){
        double error = DBL_MAX;
        float thresh = 0.0f;

        binary_classify_error(posFeats + i * posSize, posWeights, posSize, negFeats + i * negSize, negWeights, negSize, rate, thresh, error);

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
    double *pws;
    int posSize;

    float *residuals;

    int *negIdxs;
    double *nws;
    int negSize;
} NodePair;


static void split(float *posFeats, float *negFeats, int16_t thresh, NodePair *ppair, NodePair *lpair, NodePair *rpair){
    lpair->posSize = 0;
    lpair->negSize = 0;
    rpair->posSize = 0;
    rpair->negSize = 0;

    for(int i = 0; i < ppair->posSize; i++){
        if(posFeats[i] <= thresh){
            lpair->posIdxs[lpair->posSize] = ppair->posIdxs[i];
            lpair->pws[lpair->posSize] = ppair->pws[i];
            lpair->residuals[lpair->posSize] = ppair->residuals[i];

            lpair->posSize++;
        }
        else{
            rpair->posIdxs[rpair->posSize] = ppair->posIdxs[i];
            rpair->pws[rpair->posSize] = ppair->pws[i];
            rpair->residuals[rpair->posSize] = ppair->residuals[i];

            rpair->posSize++;
        }
    }

    for(int i = 0; i < ppair->negSize; i++){
        if(negFeats[i] <= thresh){
            lpair->negIdxs[lpair->negSize] = ppair->negIdxs[i];
            lpair->nws[lpair->negSize] = ppair->nws[i];

            lpair->negSize++;
        }
        else {
            rpair->negIdxs[rpair->negSize] = ppair->negIdxs[i];
            rpair->nws[rpair->negSize] = ppair->nws[i];

            rpair->negSize++;
        }
    }

    int minLeafSize = 10;

    assert(lpair->posSize > minLeafSize && lpair->negSize > minLeafSize);
    assert(rpair->posSize > minLeafSize && rpair->negSize > minLeafSize);
}


void normalize_scores(Node *leafs, int leafSize, float factor){
    float minScore = FLT_MAX, maxScore = -FLT_MAX;
    float center;

    for(int i = 0; i < leafSize; i++){
        minScore = HU_MIN(minScore, leafs[i].score);
        maxScore = HU_MAX(maxScore, leafs[i].score);
    }

    center = (minScore + maxScore) * 0.5f;

    float scale = factor / (maxScore - center);

    for(int i = 0; i < leafSize; i++)
        leafs[i].score *= scale;
}


float train(Tree *root, int depth, SampleSet *posSet, SampleSet *negSet, float radius, float prob, int pntIdx, float recall){
    cv::RNG rng(cv::getTickCount());

    int WINW = posSet->WINW;
    int WINH = posSet->WINH;

    int posSize = posSet->ssize * 0.9f;
    int negSize = negSet->ssize * 0.9f;

    int nlSize   = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);

    int nodeSize = nlSize - leafSize;

    float rates[32];

    float *posFeats = new float[posSize * FEATURE_NUM];
    float *negFeats = new float[negSize * FEATURE_NUM];

    float *bestPosFeats = new float[posSize];
    float *bestNegFeats = new float[negSize];

    float *residuals = NULL;
    double *pws = NULL, *nws = NULL;
    int *pIdxs = NULL, *nIdxs = NULL;

    FeatType *featTypes = new FeatType[FEATURE_NUM];

    NodePair *pairs = new NodePair[nlSize];

    memset(pairs, 0, sizeof(NodePair) * nlSize);

    init_rates(rates, depth);

    //printf("RANDOM SAMPLES\n");
    random_samples(posSet);
    random_samples(negSet);

    for(int i = 0; i < nlSize; i++){
        pairs[i].posIdxs = new int[posSize];
        pairs[i].pws = new double[posSize];

        pairs[i].residuals = new float[posSize];

        pairs[i].negIdxs = new int[negSize];
        pairs[i].nws = new double[negSize];
    }

    //root
    root->pw = 0;
    pws = pairs[0].pws;
    pIdxs = pairs[0].posIdxs;

    for(int i = 0; i < posSize; i++){
        float score = posSet->samples[i]->score;
        pws[i] = exp(-score);
        root->pw += pws[i];

        pIdxs[i] = i;
    }

    for(int i = 0; i < posSize; i++)
        pws[i] /= root->pw;
    root->pw = 1;

    root->nw = 0;
    nws = pairs[0].nws;
    nIdxs = pairs[0].negIdxs;

    for(int i = 0; i < negSize; i++){
        float score = negSet->samples[i]->score;
        nws[i] = exp(score);
        root->nw += nws[i];

        nIdxs[i] = i;
    }

    for(int i = 0; i < negSize; i++)
        nws[i] /= root->nw;
    root->nw = 1;

    pairs[0].posSize = posSize;
    pairs[0].negSize = negSize;

    calculate_residuals(posSet, pairs[0].posIdxs, posSize, pntIdx, pairs[0].residuals);

    for(int i = 0; i < nodeSize; i++){
        Node *node = root + i;
        int idL = i * 2 + 1;
        int idR = i * 2 + 2;

        float rate = rates[i];

        double bError;

        posSize = pairs[i].posSize;
        pIdxs = pairs[i].posIdxs;

        negSize = pairs[i].negSize;
        nIdxs = pairs[i].negIdxs;

        residuals = pairs[i].residuals;

        //printf("node: %d %d\n", posSize, negSize);

        pws = pairs[i].pws;
        nws = pairs[i].nws;

        node->posSize = pairs[i].posSize;
        node->negSize = pairs[i].negSize;

        node->flag = rng.uniform(0.0, 1.0) < prob;
        //node->flag = 1;

        node->pw = 0;
        node->nw = 0;

        for(int j = 0; j < posSize; j++)
            node->pw += pws[j];

        for(int j = 0; j < negSize; j++)
            node->nw += nws[j];

        node->score = (float)(log((node->pw + DBL_EPSILON) / (node->nw + DBL_EPSILON)) / CONF_NORM_FACTOR);

        bError = FLT_MAX;

        for(int k = 0; k < FEATURE_TIME; k++){
            ClassPair cpair;

            gen_feature_type(posSet->meanShape, FEATURE_NUM, posSet->ptsSize, radius, WINW, WINH, featTypes);

            extract_features(posSet, pIdxs, posSize, featTypes, FEATURE_NUM, posFeats);
            extract_features(negSet, nIdxs, negSize, featTypes, FEATURE_NUM, negFeats);

            if(node->flag)
                class_error(posFeats, pws, posSize, negFeats, nws, negSize, FEATURE_NUM, rate, &cpair);
            else
                class_residual(posFeats, posSize, residuals, FEATURE_NUM, rate, &cpair);

            assert(cpair.bIdx != -1);

            if(cpair.bError < bError){
                bError = cpair.bError;

                node->thresh = cpair.bThresh;
                node->featType = featTypes[cpair.bIdx];
                memcpy(bestPosFeats, posFeats + cpair.bIdx * posSize, sizeof(float) * posSize);
                memcpy(bestNegFeats, negFeats + cpair.bIdx * negSize, sizeof(float) * negSize);
            }
        }

        split(bestPosFeats, bestNegFeats, node->thresh, pairs + i, pairs + idL, pairs + idR);

        node->left = root + idL;
        node->right = root + idR;
    }

    delete [] bestPosFeats;
    delete [] bestNegFeats;
    delete [] posFeats;
    delete [] negFeats;
    delete [] featTypes;

    for(int i = nodeSize; i < nlSize; i++){
        Node *leaf = root + i;
        NodePair *pair = pairs + i;

        leaf->pw = 0.0f, leaf->nw = 0.0f;

        for(int j = 0; j < pair->posSize; j++)
            leaf->pw += pair->pws[j];

        for(int j = 0; j < pair->negSize; j++)
            leaf->nw += pair->nws[j];

        leaf->score = log((leaf->pw + 0.000001) / (leaf->nw + 0.000001)) / CONF_NORM_FACTOR;

        leaf->posSize = pair->posSize;
        leaf->negSize = pair->negSize;
        leaf->leafID = i - nodeSize;
        leaf->left = NULL;
        leaf->right = NULL;
    }

    //normalize_scores(root + nodeSize, leafSize, 1.0f);

    for(int i = 0; i < nlSize; i++){
        delete [] pairs[i].negIdxs;
        delete [] pairs[i].posIdxs;
        delete [] pairs[i].pws;
        delete [] pairs[i].nws;
        delete [] pairs[i].residuals;
    }

    delete [] pairs;

    float *scores = new float[posSet->ssize];

    for(int i = 0; i < posSet->ssize; i++){
        Sample *sample = posSet->samples[i];
        uint8_t leafID;
        sample->score += predict(root, sample->img, sample->stride, sample->curShape, sample->arg, leafID);
        scores[i] = sample->score;
    }

    sort_arr_float(scores, posSet->ssize);

    float minScore = scores[int((1.0f - recall) * posSet->ssize)] - 0.000001f;
    float maxScore = scores[posSet->ssize - 1];
    float thresh = minScore;

    printf("score: %f %f \n", minScore, maxScore);

    delete [] scores;

    for(int i = 0; i < negSet->ssize; i++){
        Sample *sample = negSet->samples[i];
        uint8_t leafID;
        sample->score += predict(root, sample->img, sample->stride, sample->curShape, sample->arg, leafID);
    }

    return thresh;
}


float predict(Tree *root, uint8_t *img, int stride, Shape &shape, uint8_t &leafID){
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    while(root->left != NULL)
        root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    leafID = root->leafID;

    return root->score;
}


float predict(Tree *root, uint8_t *img, int stride, Shape &curShape, TranArgs &arg, uint8_t &leafID){
    root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);
    root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);
    root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);

    while(root->left != NULL)
        root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);

    leafID = root->leafID;

    return root->score;
}


void save(FILE *fout, int depth, Tree *root){
    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;
    int ret;

    assert(root != NULL && fout != NULL);

    for(int i = 0; i < nodeSize; i++){
        Node *node = root + i;

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

    float scores[64];
    uint16_t values[64];
    int len = (1 << 16) - 1;
    float minv = FLT_MAX, maxv = -FLT_MAX, step;

    Node *leafs = root + nodeSize;

    for(int i = 0; i < leafSize; i++){
        scores[i] = leafs[i].score;
        minv = HU_MIN(scores[i], minv);
        maxv = HU_MAX(scores[i], maxv);
    }

    step = (maxv - minv) / len;

    for(int i = 0; i < leafSize; i++)
        values[i] = uint16_t((scores[i] - minv) / step);

    ret = fwrite(&minv, sizeof(float), 1, fout); assert(ret == 1);
    ret = fwrite(&step, sizeof(float), 1, fout); assert(ret == 1);
    ret = fwrite(values, sizeof(uint16_t), leafSize, fout);

    assert(ret == leafSize);
}


void load(FILE *fin, int depth, Tree *root){
    assert(fin != NULL && root != NULL);

    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;
    int ret;

    for(int i = 0; i < nodeSize; i++){
        Node *node = root + i;
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

    uint16_t values[64];
    float minv, step;
    Node *leafs = root + nodeSize;

    ret = fread(&minv, sizeof(float), 1, fin); assert(ret == 1);
    ret = fread(&step, sizeof(float), 1, fin); assert(ret == 1);
    ret = fread(values, sizeof(uint16_t), leafSize, fin);
    assert(ret == leafSize);

    for(int i = 0; i < leafSize; i++){
        leafs[i].score = minv + values[i] * step;
        leafs[i].leafID = i;
        leafs[i].left = NULL;
        leafs[i].right = NULL;
    }
}


void print_tree(FILE *fout, Tree *root, int depth){
    static int TREE_ID = 0;

    int nlSize   = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);

    int nodeSize = nlSize - leafSize;

    fprintf(fout, "TREE: %d\n", TREE_ID++);
    for(int i = 0; i < nodeSize; i++){
        Node *node = root + i;

        fprintf(fout, "Node: %2d, type: %d, thresh: %3d, ", i, node->flag, node->thresh);
        fprintf(fout, "posw: %.4f, poss: %6d, ", node->pw, node->posSize);
        fprintf(fout, "negw: %.4f, negs: %6d\n", node->nw, node->negSize);
    }

    for(int i = 0; i < leafSize; i++){
        Node *leaf = root + i + nodeSize;

        fprintf(fout, "Leaf: %2d, score: %5.2f, ", i, leaf->score);
        fprintf(fout, "posw: %.4f, poss: %6d, ", leaf->pw, leaf->posSize);
        fprintf(fout, "negw: %.4f, negs: %6d\n", leaf->nw, leaf->negSize);
    }
}


void release(Tree **root){
    if(*root != NULL){
        delete [] *root;
    }

    *root = NULL;
}
