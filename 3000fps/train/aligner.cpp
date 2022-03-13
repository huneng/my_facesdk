#include "aligner.h"

#define RADIUS 0.8
#define TREE_DEPTH 4
#define TREE_SIZE 4040

#define MIRROR 1
#define TIMES 1


static void initialize_aligner(Aligner **raligner, Shape &meanShape, int stage){
    char filePath[256], command[256];
    int ret;
    Aligner *aligner = NULL;
    int leafSize = 1 << (TREE_DEPTH - 1);
    int ptsSize = meanShape.ptsSize;
    int ptsSize2  = ptsSize << 1;

    for(int s = stage - 1; s >= 0; s--){
        sprintf(filePath, "cascade_%d.dat", s);

        if(load(filePath, &aligner) == 0)
            break;
    }

    if(aligner == NULL){
        aligner = new Aligner;
        memset(aligner, 0, sizeof(Aligner));

        aligner->meanShape = meanShape;
        aligner->ptsSize = meanShape.ptsSize;
    }

    if(aligner->stage == stage){
        return ;
    }
    else {
        Forest **forests = new Forest*[stage];

        if(aligner->stage > 0){
            for(int i = 0; i < aligner->stage; i++){
                forests[i] = aligner->forests[i];
            }

            delete [] aligner->forests;
        }
        for(int i = aligner->stage; i < stage; i++){
            forests[i] = new Forest;

            memset(forests[i], 0, sizeof(Forest));
        }

        aligner->forests = forests;
    }

    ret = system("mkdir -p model");
    ret = system("mkdir -p log");

    *raligner = aligner;
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


int get_direction(float *shape, int ptsSize){
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    float mx = 0, cx = 0;

    for(int i = 0; i < ptsSize; i++){
        float x = shape[i];
        float y = shape[i + ptsSize];

        minx = HU_MIN(minx, x);
        maxx = HU_MAX(maxx, x);

        mx += x;
    }

    mx /= ptsSize;
    cx = 0.5f * (minx + maxx);

    if(mx < cx ) return 0;

    return 1;
}


static float feature_distance(float *feats1, float *feats2, int size){
    float value = 0;

    for(int i = 0; i < size; i++)
        value += (feats1[i] != feats2[i]);

    printf("%f\n", value / size);
    return value / size;
}


void generate_transform_samples(Aligner *aligner, SampleSet *set, int times, SampleSet **dset){
    int WINW = set->WINW;
    int WINH = set->WINH;
    int AREA = WINW * WINH;

    int ptsSize = set->ptsSize;

    Shape meanShape;

    SampleSet *nset;

    meanShape = set->meanShape;

    for(int p = 0; p < ptsSize; p++){
        meanShape.pts[p] += (WINW >> 1);
        meanShape.pts[p + ptsSize] += (WINH >> 1);
    }

    nset = new SampleSet;

    nset->WINW = WINW;
    nset->WINH = WINH;
    nset->ptsSize = ptsSize;
    nset->ssize = set->ssize * times;
    nset->samples = new Sample*[nset->ssize];
    nset->meanShape = meanShape;

    cv::RNG rng(cv::getTickCount());
    float RAND_LEN = 0.1 * SAMPLE_IMAGE_SIZE;
    int finished = 0;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        for(int t = 0; t < times; t++){
            TranArgs arg;

            Sample *nsample;

            uint8_t *img = new uint8_t[AREA];
            Shape oriShape, curShape;
            HPoint2f center;
            HRect rect;

            memcpy(img, sample->img, sizeof(uint8_t) * AREA);
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
            //show_shape(img, WINW, WINH, WINW, curShape, "cur");
            //show_shape(img, WINW, WINH, WINW, oriShape, "ori");

            similarity_transform(curShape, meanShape, arg);
            arg.cen2.x = WINW >> 1;
            arg.cen2.y = WINH >> 1;

            affine_sample(img, WINW, WINH, WINW, curShape, arg);
            affine_shape(oriShape, oriShape, arg);

            nsample = new Sample;
            nsample->img = img;
            nsample->oriShape = oriShape;
            nsample->curShape = curShape;
            nsample->stride = WINW;

            nset->samples[i * times + t] = nsample;
        }

#pragma omp critical
        {
            finished ++;
            printf("%d\r", finished), finished++;
        }
    }

    *dset = nset;
}


void validate(Aligner *aligner, SampleSet *set){
    SampleSet *nset;

    int ptsSize2 = set->ptsSize << 1;
    float *rsdls;

    generate_transform_samples(aligner, set, 1, &nset);

    rsdls = new float[nset->ssize * ptsSize2];

    for(int i = 0; i < nset->ssize; i++){
        Sample *sample = nset->samples[i];
        float *ptrRsdl = rsdls + i * ptsSize2;

        TranArgs arg;
        Shape curShape;
        Shape oriShape;

        curShape = sample->curShape;
        oriShape = sample->oriShape;

        similarity_transform(curShape, set->meanShape, arg);

        arg.cen2 = arg.cen1;

        affine_shape(curShape, curShape, arg);
        affine_shape(oriShape, oriShape, arg);

        for(int p = 0; p < ptsSize2; p++)
            ptrRsdl[p] = oriShape.pts[p] - curShape.pts[p];

        printf("%d\r", i), fflush(stdout);
    }

    mean_residual(rsdls, nset->ssize, nset->ptsSize);

    release(&nset);

    delete [] rsdls;
}


void train_aligner(const char *listFile, int stage, Aligner **resAligner){
    Aligner *aligner = NULL;

    SampleSet *set = NULL;

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

    printf("INITIALIZE ALIGNER\n");
    initialize_aligner(&aligner, set->meanShape, stage);

    printf("SAMPLE: %d, TIMES: %d, TREE SIZE: %d, DEPTH: %d, PTS SIZE: %d, STAGE: %d\n", set->ssize, TIMES, TREE_SIZE, TREE_DEPTH, aligner->meanShape.ptsSize, stage);

    for(int s = aligner->stage; s < stage; s++){
        printf("---------------------- CASCADE %d ---------------------------\n", s);

        Forest *forest = aligner->forests[s];
        SampleSet *nset;

        printf("GENERATE TRANSFROM SAMPLES\n");
        generate_transform_samples(aligner, set, TIMES, &nset);
        release_data(set);

        printf("TRAIN FOREST\n");
        //train_forest(nset, TREE_SIZE, TREE_DEPTH, RADIUS * powf(0.8, s), aligner->forests[s]);
        train_forest(nset, TREE_SIZE, TREE_DEPTH, RADIUS, aligner->forests[s]);
        release(&nset);

        aligner->stage++;

        printf("SAVE MODEL\n");
        sprintf(outPath, "model/cascade_%d.dat", s);
        save(outPath, aligner);

        printf("LOAD SAMPLES\n");
        ret = load(SAMPLE_FILE, &set);assert(ret == 0);

        printf("VALIDATE\n");
        validate(aligner, set);

        printf("---------------------------------------------------------\n\n");
    }

    *resAligner = aligner;
}


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, HRect &srect, Shape &curShape){
    TranArgs arg;
    HRect rect;

    float scale;

    int ptsSize = aligner->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int winSize = SAMPLE_IMAGE_SIZE * 2 + 1;


    uint8_t *patch, *src;

    assert(aligner != NULL && img != NULL);

    rect.x = srect.x - (srect.width  >> 1);
    rect.y = srect.y - (srect.height >> 1);
    rect.width  = srect.width << 1;
    rect.height = srect.height << 1;

    patch = new uint8_t[rect.width * rect.height + winSize * winSize];
    src = patch + rect.width * rect.height;

    extract_area_from_image(img, width, height, stride, patch, rect);

    scale = (float)rect.width / winSize;

    resizer_bilinear_gray(patch, rect.width, rect.height, rect.width, src, winSize, winSize, winSize);

    curShape = aligner->meanShape;

    for(int p = 0; p < ptsSize2; p++)
        curShape.pts[p] += SAMPLE_IMAGE_SIZE;

    for(int s = 0; s < aligner->stage; s++){
        Forest *forest = aligner->forests[s];

        memcpy(patch, src, sizeof(uint8_t) * winSize * winSize);

        if(s > 0){
            similarity_transform(curShape, aligner->meanShape, arg);
            arg.cen2.x = winSize >> 1;
            arg.cen2.y = winSize >> 1;

            affine_sample(patch, winSize, winSize, winSize, curShape, arg);

            predict(forest, patch, winSize, winSize, winSize, curShape);

            arg.scale = 1.0f / arg.scale;
            arg.angle = -arg.angle;

            HU_SWAP(arg.cen1, arg.cen2, HPoint2f);
            affine_shape(curShape, curShape, arg);
        }
        else
            predict(forest, patch, winSize, winSize, winSize, curShape);

    }

    for(int i = 0, j = ptsSize; i < ptsSize; i++, j++){
        curShape.pts[i] = curShape.pts[i] * scale + rect.x;
        curShape.pts[j] = curShape.pts[j] * scale + rect.y;
    }

    delete [] patch;
}


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, Shape &curShape){
    HRect rect;

    rect = get_shape_rect(curShape);

    predict(aligner, img, width, height, stride, rect, curShape);
}


int save(const char *filePath, Aligner *aligner){
    if(aligner == NULL || filePath == NULL)
        return 1;

    FILE *fp = fopen(filePath, "wb");

    if(fp == NULL){
        printf("Can't save file %s\n", filePath);
        return 2;
    }

    int ret;

    ret = fwrite(&aligner->stage, sizeof(int), 1, fp); assert(ret == 1);
    ret = fwrite(&aligner->ptsSize, sizeof(int), 1, fp); assert(ret == 1);

    ret = fwrite(&aligner->meanShape, sizeof(Shape), 1, fp); assert(ret == 1);

    for(int i = 0; i < aligner->stage; i++){
        save(fp, aligner->forests[i]);
    }

    fclose(fp);
    return 0;
}



int load(const char *filePath, Aligner **raligner){
    if(filePath == NULL)
        return 1;

    FILE *fp = fopen(filePath, "rb");

    if(fp == NULL){
        printf("Can't save file %s\n", filePath);
        return 2;
    }

    int ret;
    Aligner *aligner = new Aligner;

    ret = fread(&aligner->stage, sizeof(int), 1, fp); assert(ret == 1);
    ret = fread(&aligner->ptsSize, sizeof(int), 1, fp); assert(ret == 1);

    ret = fread(&aligner->meanShape, sizeof(Shape), 1, fp); assert(ret == 1);

    aligner->forests = new Forest*[aligner->stage];

    for(int i = 0; i < aligner->stage; i++){
        load(fp, &aligner->forests[i]);
    }

    fclose(fp);

    *raligner = aligner;

    return 0;
}


void release_data(Aligner *aligner){
    if(aligner->stage == 0 || aligner->forests == NULL)
        return;

    for(int i = 0; i < aligner->stage; i++){
        release(&aligner->forests[i]);
    }

    delete [] aligner->forests;

    aligner->stage = 0;
    aligner->forests = NULL;
}


void release(Aligner **aligner){
    if(*aligner == NULL)
        return;

    release_data(*aligner);
    delete *aligner;
    *aligner = NULL;
}
