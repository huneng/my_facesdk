#include "forest.h"
#include "linear.h"

int generate_negative_images(const char *listFile, const char *outfile){
    FILE *fin = fopen(outfile, "rb");
    if(fin != NULL){
        fclose(fin);
        return 0;
    }

    std::vector<std::string> imgList;
    read_file_list(listFile, imgList);

    int size = imgList.size();

    printf("GENERATE NEGATIVE IMAGES %ld\n", imgList.size());
    FILE *fout = fopen(outfile, "wb");
    if(fout == NULL){
        printf("Can't open file %s\n", outfile);
        return 1;
    }

    char rootDir[256], fileName[256], ext[30];
    int ret;

    ret = fwrite(&size, sizeof(int), 1, fout), assert(ret == 1);

    for(int i = 0; i < size; i++){
        const char *imgPath = imgList[i].c_str();
        cv::Mat img = cv::imread(imgPath, 0);

        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            continue;
        }

        analysis_file_path(imgPath, rootDir, fileName, ext);

        ret = fwrite(fileName, sizeof(char), 255, fout); assert(ret == 255);

        if(img.cols > img.rows && img.rows > 720)
            cv::resize(img, img, cv::Size(img.cols * 720 / img.rows, 720));
        else if(img.rows > img.cols && img.cols > 720)
            cv::resize(img, img, cv::Size(720, 720 * img.rows / img.cols));

        if(img.rows * img.cols > 1024 * 1024){
            float scale = 1024.0f / img.rows;
            scale = HU_MIN(1024.0f / img.cols, scale);
            cv::resize(img, img, cv::Size(scale * img.cols, scale * img.rows));
        }

        assert(img.cols * img.rows <= 1024 * 1024);

        ret = fwrite(&img.rows, sizeof(int), 1, fout); assert(ret == 1);
        ret = fwrite(&img.cols, sizeof(int), 1, fout); assert(ret == 1);

        if(img.cols == img.step)
            ret = fwrite(img.data, sizeof(uint8_t), img.rows * img.cols, fout);
        else
            for(int y = 0; y < img.rows; y++)
                ret = fwrite(img.data + y * img.step, sizeof(uint8_t), img.cols, fout);
        if(i % 10000 == 0)
            printf("%d ", i), fflush(stdout);
    }

    printf("\n");
    fclose(fout);

    printf("IMAGE SIZE: %d\n", size);
    return 0;
}


void load_images(FILE *fin, uint8_t **imgs, int *widths, int *heights, char **fileNames, int size, int tflag){
    int ret = 0;

    uint8_t *buffer = new uint8_t[4096 * 4096];

    for(int j = 0; j < size; j++){
        int width, height;

        ret = fread(fileNames[j], sizeof(char), 255, fin); assert(ret == 255);
        ret = fread(heights + j, sizeof(int), 1, fin); assert(ret == 1);
        ret = fread(widths + j, sizeof(int), 1, fin); assert(ret == 1);

        ret = fread(imgs[j], sizeof(uint8_t), widths[j] * heights[j], fin); assert(ret == widths[j] * heights[j]);

        if(tflag)
            transform_image(imgs[j], widths[j], heights[j], widths[j], buffer);

        /*
        cv::Mat sImg(heights[j], widths[j], CV_8UC1, imgs[j]);
        cv::imshow("sImg", sImg);
        cv::waitKey();
        //*/
    }

    delete [] buffer;
}


int detect_image(uint8_t *img, int width, int height, int stride, char *fileName,
        NegGenerator *generator, uint8_t *buffer, uint8_t *meanBuffer, SampleSet *set, int maxSize){
    int count = 0;
    float scale = 1.0f;

    int WINW = set->WINW;
    int WINH = set->WINH;

    int dx = generator->dx;
    int dy = generator->dy;

    for(int l = 0; l < 20; l++){
        int w, h, s;

        if(width > height){
            h = WINH * scale;
            w = h * width / height;
            w = HU_MAX(w, WINW);
        }
        else {
            w = WINW * scale;
            h = w * height / width;
            h = HU_MAX(h, WINH);
        }

        if(w * h > 4096 * 4096) break;

        resizer_bilinear_gray(img, width, height, stride, buffer, w, h, w);
        //mean_filter_3x3(buffer, w, h, w, meanBuffer);

        /*
        cv::Mat sImg(h, w, CV_8UC1, buffer, w);
        cv::imshow("sImg2", sImg);
        cv::waitKey();
        //*/
        s = w;
        w -= WINW;
        h -= WINH;

        for(int y = 0; y <= h; y += dy){
            for(int x = 0; x <= w; x += dx){
                uint8_t *patch = buffer + y * s + x;
                Shape curShape;
                float score;

                argument(set->meanShape, curShape);
                //curShape = set->meanShape;

                int ret = predict(generator->forests, generator->fsize, set->meanShape, patch, s, curShape, score);
                if(ret == 1){
                    Sample *sample = new Sample;

                    memset(sample, 0, sizeof(Sample));

                    sample->oriShape = set->meanShape;
                    sample->curShape = curShape;

                    for(int p = 0; p < set->meanShape.ptsSize; p++){
                        sample->curShape.pts[p].x += x;
                        sample->curShape.pts[p].y += y;

                        sample->oriShape.pts[p].x += x;
                        sample->oriShape.pts[p].y += y;
                    }

                    extract_sample_from_image(buffer, w + WINW, h + WINW, s, set->meanShape, set->WINW, set->WINH, BORDER_FACTOR * set->WINW, sample);

                    similarity_transform(set->meanShape, sample->curShape, sample->arg);

                    sample->score = score;

                    sprintf(sample->patchName, "%s_%d.jpg", fileName, count);
                    count++;

                    add_sample_capacity_unchange(set, sample);
                }
            }
        }

        if(count > maxSize)
            break;

        scale *= 1.13f;
    }

    return count;
}


void generate_negative_samples(SampleSet *negSet, NegGenerator *generator, int needSize){
    if(needSize <= 0)
        return;

    int WINW = negSet->WINW;
    int WINH = negSet->WINH;

    const int BUF_SIZE = 1000;

    uint8_t **imgs;
    char **fileNames;
    int *widths, *heights;

    imgs = new uint8_t*[BUF_SIZE];
    fileNames = new char*[BUF_SIZE];
    widths = new int[BUF_SIZE];
    heights = new int[BUF_SIZE];

    for(int i = 0; i < BUF_SIZE; i++){
        fileNames[i] = new char[256];
        imgs[i] = new uint8_t[1024 * 1024];
    }

    int threadNum = omp_get_num_procs() - 1;

    SampleSet *sets = new SampleSet[threadNum];
    uint8_t **sImgs = new uint8_t*[threadNum];
    uint8_t **mImgs = new uint8_t*[threadNum];

    memset(sets, 0, sizeof(SampleSet) * threadNum);

    for(int i = 0; i < threadNum; i++){
        sImgs[i] = new uint8_t[4096 * 4096];
        mImgs[i] = new uint8_t[4096 * 4096];

        sets[i].WINW = negSet->WINW;
        sets[i].WINH = negSet->WINH;
        sets[i].meanShape = negSet->meanShape;
        sets[i].ptsSize = negSet->ptsSize;
        sets[i].ssize = 0;

        reserve(sets + i, (needSize / threadNum + 1) * 2.0f);
    }

    int id = generator->id;
    int ret;
    int ssize;

    //printf("%d %d %d\n", generator->isize, generator->id, generator->fsize);
    while(1){
        printf("%d ", id);
        int ei = HU_MIN(BUF_SIZE, generator->isize - id);

        //printf("Load images %d, ", ei);fflush(stdout);
        load_images(generator->fin, imgs, widths, heights, fileNames, ei, generator->tflag);

        //printf("sample, "); fflush(stdout);
#pragma omp parallel for num_threads(threadNum)
        for(int i = 0; i < ei; i++){
            int tId = omp_get_thread_num();
            uint8_t *img = imgs[i];

            int width = widths[i];
            int height = heights[i];

            char *fileName = fileNames[i];

            uint8_t *sImg = sImgs[tId];
            uint8_t *mImg = mImgs[tId];

            SampleSet *set = sets + tId;

            int count = detect_image(img, width, height, width, fileName, generator, sImg, mImg, set, generator->maxCount);
        }

        ssize = 0;
        for(int i = 0; i < threadNum; i++){
            ssize += sets[i].ssize;
        }

        id += BUF_SIZE;

        if(id >= generator->isize){
            id = 0;
            fclose(generator->fin);
            generator->fin = fopen(generator->filePath, "rb");
            ret = fread(&generator->isize, sizeof(int), 1, generator->fin);
            assert(ret == 1);
        }

        printf("%d, ", ssize);fflush(stdout);

        if(ssize > needSize)
            break;
    }
    printf("\n");

    generator->id = id;

    ssize += negSet->ssize;
    reserve(negSet, ssize);

    for(int i = 0; i < threadNum; i++){
        memcpy(negSet->samples + negSet->ssize, sets[i].samples, sizeof(Sample*) * sets[i].ssize);
        negSet->ssize += sets[i].ssize;

        delete [] sets[i].samples;

        sets[i].samples = NULL;
        sets[i].ssize = 0;
        sets[i].capacity = 0;

        delete [] sImgs[i];
        delete [] mImgs[i];
    }

    delete [] sets;
    delete [] sImgs;
    delete [] mImgs;

    for(int i = 0; i < BUF_SIZE; i++){
        delete [] imgs[i];
        delete [] fileNames[i];
    }

    delete [] imgs;
    delete [] fileNames;
    delete [] widths;
    delete [] heights;
}


void global_regress(Forest *forest, SampleSet *posSet);


void print_recall_and_precision(SampleSet *posSet, SampleSet *negSet, float thresh){
    float TP = 0, FP = 0;
    float FN = 0, TN = 0;

    for(int i = 0; i < posSet->ssize; i++){
        Sample* sample = posSet->samples[i];

        if(sample->score <= thresh)
            FN++;
        else
            TP ++;
    }


    for(int i = 0; i < negSet->ssize; i++){
        Sample* sample = negSet->samples[i];
        if(sample->score <= thresh)
            TN ++;
        else
            FP ++;
    }

    TP /= posSet->ssize;
    FN /= posSet->ssize;

    TN /= negSet->ssize;
    FP /= negSet->ssize;

    printf("recall: %f, precision: %f, thresh: %f\n", TP, TP / (TP + FP), thresh);
}


void train(Forest *forest, SampleSet *posSet, SampleSet *negSet, TrainParams *params, NegGenerator *generator){
    static int FOREST_ID = 0;

    int ptsSize = posSet->ptsSize;
    int ptsSize2 = ptsSize << 1;

    printf("UPDATE TRANSLATE ARGUMENTS\n");
    reset_scores_and_args(posSet);
    reset_scores_and_args(negSet);

    forest->treeSize = 0;
    forest->depth = params->depth;
    forest->trees = new Tree*[params->treeSize];
    forest->threshes = new float[params->treeSize];
    forest->offsets = NULL;

    int nlSize = (1 << forest->depth) - 1;

    FILE *fout = fopen("log/classifier.txt", "w");

    assert(omp_get_num_procs() > 1);

    for(int i = 0; i < params->treeSize; i++){
        if(i == 150) params->recall = 1.0f;
        printf("classifier: %d\n", i);
        Tree *tree = new Node[nlSize];

        int needSize = params->npRate * posSet->ssize - negSet->ssize;

        printf("GENERATE NEGATIVE SAMPLES %d\n", needSize);
        generate_negative_samples(negSet, generator, needSize);
        //write_images(negSet, "log/neg", negSet->ssize / 4000 + 1);
        //exit(0);

        assert(negSet->ssize >= 0.25 * posSet->ssize);

        forest->threshes[i] = train(tree, forest->depth, posSet, negSet, params->radius, params->prob, i % ptsSize2, params->recall);

        if(params->flag){
            //refine_samples(posSet, forest->threshes[i], 0);
            refine_samples(negSet, forest->threshes[i], 1);
            printf("posSize: %d, negSize: %d, thresh: %f\n", posSet->ssize, negSet->ssize, forest->threshes[i]);
        }
        else{
            print_recall_and_precision(posSet, negSet, forest->threshes[i]);
        }

        forest->trees[i] = tree;
        forest->treeSize++;

        print_tree(fout, forest->trees[i], forest->depth); fflush(fout);

        printf("\n");
    }

    fclose(fout);

    if(params->flag == 0){
        refine_samples(negSet, forest->threshes[forest->treeSize - 1], 1);
        write_images(negSet, "log/neg", 1);
    }

    release_data(negSet);

    char command[256];
    sprintf(command, "mv log/classifier.txt log/classifier_%d.txt", FOREST_ID++);
    int ret = system(command);

    double *rsdlsX, *rsdlsY;

    rsdlsX = new double[posSet->ptsSize * posSet->ssize];
    rsdlsY = new double[posSet->ptsSize * posSet->ssize];

    printf("CALCULATE RESIDUALS\n");
    calculate_residuals(posSet, rsdlsX, rsdlsY);

    delete [] rsdlsX;
    delete [] rsdlsY;

    printf("GLOBAL REGRESS\n");
    global_regress(forest, posSet);

    for(int i = 0; i < posSet->ssize; i++){
        Sample *sample = posSet->samples[i];

        int ret = predict(forest, posSet->meanShape, sample->img, sample->stride, sample->curShape, sample->score);
    }

    rsdlsX = new double[posSet->ptsSize * posSet->ssize];
    rsdlsY = new double[posSet->ptsSize * posSet->ssize];

    printf("CALCULATE RESIDUALS\n");
    calculate_residuals(posSet, rsdlsX, rsdlsY);

    delete [] rsdlsX;
    delete [] rsdlsY;
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
        params[i].eps = 0.00001;
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


int write_data(const char *filePath, float *feats, float *rsdls, int ssize, int featDim, int ptsSize2){
    FILE *fout = fopen(filePath, "wb");

    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;

    ret = fwrite(&ssize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&featDim, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&ptsSize2, sizeof(int), 1, fout); assert(ret == 1);

    ret = fwrite(feats, sizeof(float), ssize * featDim, fout); assert(ret == ssize * featDim);
    ret = fwrite(rsdls, sizeof(float), ssize * ptsSize2, fout); assert(ret == ssize * ptsSize2);

    fclose(fout);

    return 0;
}


void generate_binary_features(SampleSet *posSet, Forest *forest, float *feats){
    int WINW = posSet->WINW;
    int ssize = posSet->ssize;
    int treeSize = forest->treeSize;
    int leafSize = 1 << (forest->depth - 1);
    int featDim = leafSize * treeSize;

    memset(feats, 0, sizeof(float) * featDim * ssize);

    for(int i = 0; i < ssize; i++){
        Sample *sample = posSet->samples[i];

        float *ptrFeats = feats + i * featDim;

        for(int t = 0; t < treeSize; t++){
            uint8_t leafID = 0;
            predict(forest->trees[t], sample->img, sample->stride, sample->curShape, sample->arg, leafID);

            ptrFeats[leafID] = 1;
            ptrFeats += leafSize;
        }
    }
}


void generate_residuals(SampleSet *posSet, float *rsdls){
    int ptsSize = posSet->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int ssize = posSet->ssize;

    assert(ptsSize == 68);

    for(int i = 0; i < ssize; i++){
        Sample *sample = posSet->samples[i];

        float *ptrRsdl = rsdls + i * ptsSize2;

        TranArgs arg;

        similarity_transform(sample->curShape, posSet->meanShape, arg);
        for(int p = 0; p < ptsSize; p++){
            int j = p << 1;

            float dx = sample->oriShape.pts[p].x - sample->curShape.pts[p].x;
            float dy = sample->oriShape.pts[p].y - sample->curShape.pts[p].y;

            ptrRsdl[j]     = dx * arg.scosa + dy * arg.ssina;
            ptrRsdl[j + 1] = dy * arg.scosa - dx * arg.ssina;
        }
    }
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

        t = 0; f = 0;
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


PCAModel* train_binary_feature_pca(Forest *forest, SampleSet *posSet){
    int WINW = posSet->WINW;
    int ssize = posSet->ssize;
    int treeSize = forest->treeSize;
    int leafSize = 1 << (forest->depth - 1);
    int featDim = leafSize * treeSize;

    float *feats = new float[featDim * ssize];

    PCAModel *model = NULL;

    memset(feats, 0, sizeof(float) * featDim * ssize);

    for(int i = 0; i < ssize; i++){
        Sample *sample = posSet->samples[i];

        float *ptrFeats = feats + i * featDim;

        TranArgs arg;

        similarity_transform(posSet->meanShape, sample->oriShape, arg);

        for(int t = 0; t < treeSize; t++){
            uint8_t leafID = 0;
            predict(forest->trees[t], sample->img, sample->stride, sample->oriShape, arg, leafID);

            ptrFeats[leafID] = 1;
            ptrFeats += leafSize;
        }
    }

    model = train_pca(feats, featDim, ssize, featDim, 0.9f);
    delete [] feats;

    return model;
}



#define TRAIN_DATA "log/train_data.bin"

void global_regress(Forest *forest, SampleSet *posSet){
    int ssize = posSet->ssize;
    int ptsSize = posSet->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int leafSize = 1 << (forest->depth - 1);
    int treeSize = forest->treeSize;
    int featDim = leafSize * treeSize;
    int dim, rsdlDim;

    float *feats, *rsdls, *featsPCA, *rsdlsPCA, *coeffs;

    PCAModel *fmodel, *rmodel;

    printf("TRAIN FEATURE PCA\n");
    fmodel = train_binary_feature_pca(forest, posSet);
    dim = fmodel->dim;

    feats    = new float[ssize * featDim];
    featsPCA = new float[ssize * dim];

    printf("GENERATE BINARY FEATURES\n");
    generate_binary_features(posSet, forest, feats);

    printf("PROJECTION FEATURES\n");
#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < ssize; i++)
        pca_projection_feature(feats + i * featDim, featDim, dim, fmodel->vmean, fmodel->eigenVectors, featsPCA + i * dim);

    delete [] feats;

    rsdls    = new float[ssize * ptsSize2];
    rsdlsPCA = new float[ssize * ptsSize2];

    printf("CALCULATE RESIDUALS\n");
    generate_residuals(posSet, rsdls);

    printf("TRAIN RESIDUALS PCA MODEL\n");
    rmodel = train_pca(rsdls, ptsSize2, ssize, ptsSize2, 0.9);
    rsdlDim = rmodel->dim;

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < ssize; i++)
        pca_projection_feature(rsdls + i * ptsSize2, ptsSize2, rsdlDim, rmodel->vmean, rmodel->eigenVectors, rsdlsPCA + i * rsdlDim);

    delete [] rsdls;

    printf("WRITE DATA\n");
    write_data(TRAIN_DATA, featsPCA, rsdlsPCA, ssize, dim, rsdlDim);

    delete [] featsPCA;
    delete [] rsdlsPCA;

    coeffs = new float[dim * rsdlDim];

    printf("SOLVE PROBLE\n");
    solve_linear_equation_liblinear(TRAIN_DATA, coeffs);

    printf("SYNCHRONIZE\n");
    float *W = new float[featDim * rsdlDim];

    cv::Mat_<float> A(dim, featDim, fmodel->eigenVectors);
    cv::Mat_<float> X(dim, rsdlDim, coeffs);
    cv::Mat_<float> B(featDim, rsdlDim, W);

    B = A.t() * X;

    delete [] coeffs; coeffs = NULL;

    assert(forest->offsets == NULL);

    forest->offsets = new float*[treeSize];
    forest->rsdlDim = rsdlDim;

    for(int i = 0; i < treeSize; i++)
        forest->offsets[i] = W + i * leafSize * rsdlDim;


    write_matrix(W, rsdlDim, featDim, "log/W.txt");

    A = cv::Mat_<float>(1, featDim, fmodel->vmean);
    X = cv::Mat_<float>(featDim, rsdlDim, W);
    B = cv::Mat_<float>(1, rsdlDim, forest->bias);

    B = A * X;

    for(int i = 0; i < rsdlDim; i++)
        forest->bias[i] = -forest->bias[i];

    forest->rsdlMean = new float[ptsSize2];
    forest->rsdlCoeff = new float[rsdlDim * ptsSize2];

    memcpy(forest->rsdlMean, rmodel->vmean, sizeof(float) * ptsSize2);
    memcpy(forest->rsdlCoeff, rmodel->eigenVectors, sizeof(float) * rsdlDim * ptsSize2);

    release(&fmodel);
    release(&rmodel);

    write_matrix(forest->bias, rsdlDim, 1, "log/bias.txt");
}


#define MAX_TREE_SIZE 512

int predict_one(Forest *forest, Shape &meanShape, uint8_t *img, int stride, Shape &shape, float &score){
    score = 0;
    if(forest->treeSize == 0)
        return 1;

    uint8_t leafIDs[MAX_TREE_SIZE];
    int ptsSize = shape.ptsSize;
    int ptsSize2 = shape.ptsSize << 1;

    assert(forest->treeSize <= MAX_TREE_SIZE);

    int i = 0;
    for(; i <= forest->treeSize - 4; ){
        score += predict(forest->trees[i], img, stride, shape, leafIDs[i]);
        if(score <= forest->threshes[i])
            return 0;
        i++;

        score += predict(forest->trees[i], img, stride, shape, leafIDs[i]);
        if(score <= forest->threshes[i])
            return 0;
        i++;

        score += predict(forest->trees[i], img, stride, shape, leafIDs[i]);
        if(score <= forest->threshes[i])
            return 0;
        i++;

        score += predict(forest->trees[i], img, stride, shape, leafIDs[i]);
        if(score <= forest->threshes[i])
            return 0;
        i++;
    }

    for(; i < forest->treeSize; i ++){
        score += predict(forest->trees[i], img, stride, shape, leafIDs[i]);
        if(score <= forest->threshes[i])
            return 0;
    }


    if(forest->offsets == NULL)
        return 1;


    float buffer[MAX_PTS_SIZE * 2];
    float rsdl[MAX_PTS_SIZE * 2];

    memcpy(buffer, forest->bias, sizeof(float) * forest->rsdlDim);
    memcpy(rsdl, forest->rsdlMean, sizeof(float) * ptsSize2);

    for(int i = 0; i < forest->treeSize; i++){
        float *ptr = forest->offsets[i] + leafIDs[i] * forest->rsdlDim;

        for(int j = 0; j < forest->rsdlDim; j++)
            buffer[j] += ptr[j];
    }

    float *ptrCoeff = forest->rsdlCoeff;

    for(int d = 0; d < forest->rsdlDim; d++){
        float t = buffer[d];
        for(int p = 0; p < ptsSize2; p++)
            rsdl[p] += t * ptrCoeff[p];

        ptrCoeff += ptsSize2;
    }

    for(int p = 0; p < ptsSize; p++){
        int id = p << 1;
        shape.pts[p].x += rsdl[id];
        shape.pts[p].y += rsdl[id + 1];
    }

    return 1;
}



int predict(Forest *forest, Shape &meanShape, uint8_t *img, int stride, Shape &curShape, float &score){
    score = 0;

    if(forest->treeSize == 0)
        return 1;

    TranArgs arg;
    uint8_t leafIDs[512];
    int ptsSize = curShape.ptsSize;
    int ptsSize2 = curShape.ptsSize << 1;

    similarity_transform(meanShape, curShape, arg);

    int i = 0;
    for(; i <= forest->treeSize - 4; ){
        REPEAT_LINE_4(score += predict(forest->trees[i], img, stride, curShape, arg, leafIDs[i]);if(score <= forest->threshes[i]) return 0;i++;);
    }

    for(; i < forest->treeSize; i ++){
        score += predict(forest->trees[i], img, stride, curShape, arg, leafIDs[i]);
        if(score <= forest->threshes[i])
            return 0;
    }


    if(forest->offsets == NULL)
        return 1;

    float buffer[MAX_PTS_SIZE * 2];
    float rsdl[MAX_PTS_SIZE * 2];

    memcpy(buffer, forest->bias, sizeof(float) * forest->rsdlDim);
    memcpy(rsdl, forest->rsdlMean, sizeof(float) * ptsSize2);

    for(int i = 0; i < forest->treeSize; i++){
        float *ptr = forest->offsets[i] + leafIDs[i] * forest->rsdlDim;

        for(int j = 0; j < forest->rsdlDim; j++)
            buffer[j] += ptr[j];
    }

    float *ptrCoeff = forest->rsdlCoeff;

    for(int d = 0; d < forest->rsdlDim; d++){
        float t = buffer[d];
        for(int p = 0; p < ptsSize2; p++)
            rsdl[p] += t * ptrCoeff[p];

        ptrCoeff += ptsSize2;
    }

    float sina = arg.ssina;
    float cosa = arg.scosa;

    for(int p = 0; p < ptsSize; p++){
        int j = p << 1;
        float x = rsdl[j];
        float y = rsdl[j + 1];

        curShape.pts[p].x += (x * cosa + y * sina);
        curShape.pts[p].y += (y * cosa - x * sina);
    }

    return 1;
}


int predict(Forest *forest, int size, Shape &meanShape, uint8_t *img, int stride, Shape &shape, float &score){
    for(int i = 0; i < size; i++){
        if(predict(forest + i, meanShape, img, stride, shape, score) == 0)
            return 0;
    }

    return 1;
}


void write_offsets(FILE *fout, float *offsets, int ptsSize2, int allLeafSize){
    assert(offsets != NULL);
    assert(fout != NULL);

    uint8_t *buffer = new uint8_t[ptsSize2];
    int ret;

    for(int i = 0; i < allLeafSize; i++){
        float *ptr = offsets + i * ptsSize2;
        float maxv = ptr[0];
        float minv = ptr[0];

        for(int j = 1; j < ptsSize2; j++){
            maxv = HU_MAX(maxv, ptr[j]);
            minv = HU_MIN(minv, ptr[j]);
        }

        float step = (maxv - minv) / 255;

        for(int j = 0; j < ptsSize2; j++){
            int index = (ptr[j] - minv) / step;
            assert(index >= 0 && index < 256);
            buffer[j] = uint8_t(index);
        }

        ret = fwrite(&minv, sizeof(float), 1, fout); assert(ret == 1);
        ret = fwrite(&step, sizeof(float), 1, fout); assert(ret == 1);
        ret = fwrite(buffer, sizeof(uint8_t), ptsSize2, fout); assert(ret == ptsSize2);
    }

    delete [] buffer;
}


void read_offsets(FILE *fin, float *offsets, int ptsSize2, int allLeafSize){
    assert(fin != NULL);
    assert(offsets != NULL);

    uint8_t *buffer = new uint8_t[ptsSize2];

    for(int i = 0; i < allLeafSize; i++){
        float *ptr = offsets + i * ptsSize2;
        float minv, step;
        int ret;

        ret = fread(&minv, sizeof(float), 1, fin); assert(ret == 1);
        ret = fread(&step, sizeof(float), 1, fin); assert(ret == 1);
        ret = fread(buffer, sizeof(uint8_t), ptsSize2, fin); assert(ret == ptsSize2);

        for(int j = 0; j < ptsSize2; j++){
            ptr[j] = minv + step * buffer[j];
        }
    }

    delete [] buffer;
}


void save(FILE *fout, int ptsSize, Forest *forest){
    int ret;
    int ptsSize2 = ptsSize << 1;
    int leafSize = 1 << (forest->depth - 1);

    ret = fwrite(&forest->treeSize, sizeof(int), 1, fout);
    assert(ret == 1);

    ret = fwrite(&forest->depth, sizeof(int), 1, fout);
    assert(ret == 1);

    ret = fwrite(&forest->rsdlDim, sizeof(int), 1, fout);

    float minv = FLT_MAX, maxv = -FLT_MAX, step;
    int len = (1 << 16) - 1;

    for(int i = 0; i < forest->treeSize; i++){
        minv = HU_MIN(minv, forest->threshes[i]);
        maxv = HU_MAX(maxv, forest->threshes[i]);
    }

    step = (maxv - minv) / len;

    ret = fwrite(&minv, sizeof(float), 1, fout); assert(ret == 1);
    ret = fwrite(&step, sizeof(float), 1, fout); assert(ret == 1);

    for(int i = 0; i < forest->treeSize; i++){
        uint16_t value = (forest->threshes[i] - minv) / step;

        ret = fwrite(&value, sizeof(uint16_t), 1, fout);
        assert(ret == 1);

        save(fout, forest->depth, forest->trees[i]);
    }

    write_offsets(fout, forest->offsets[0], forest->rsdlDim, forest->treeSize * leafSize);

    ret = fwrite(forest->rsdlMean, sizeof(float), ptsSize2, fout); assert(ret == ptsSize2);
    ret = fwrite(forest->rsdlCoeff, sizeof(float), forest->rsdlDim * ptsSize2, fout); assert(ret == forest->rsdlDim * ptsSize2);
    ret = fwrite(forest->bias, sizeof(float), forest->rsdlDim, fout); assert(ret == forest->rsdlDim);
}


void load(FILE *fin, int ptsSize, Forest *forest){
    int ret;

    ret = fread(&forest->treeSize, sizeof(int), 1, fin);
    assert(ret == 1);

    ret = fread(&forest->depth, sizeof(int), 1, fin);
    assert(ret == 1);

    ret = fread(&forest->rsdlDim, sizeof(int), 1, fin);
    assert(ret == 1);

    int leafSize = 1 << (forest->depth - 1);
    int nlSize   = (1 << forest->depth) - 1;
    int ptsSize2 = ptsSize << 1;
    int len = forest->rsdlDim * leafSize;

    forest->trees = new Tree*[forest->treeSize];
    forest->threshes = new float[forest->treeSize];
    forest->offsets = new float*[forest->treeSize];
    forest->offsets[0] = new float[forest->treeSize * len];
    forest->rsdlMean = new float[ptsSize2];
    forest->rsdlCoeff = new float[forest->rsdlDim * ptsSize2];

    float minv, step;

    ret = fread(&minv, sizeof(float), 1, fin); assert(ret == 1);
    ret = fread(&step, sizeof(float), 1, fin); assert(ret == 1);

    for(int i = 0; i < forest->treeSize; i++){
        uint16_t value;

        forest->trees[i] = new Tree[nlSize];
        forest->offsets[i] = forest->offsets[0] +  i * len;

        ret = fread(&value, sizeof(uint16_t), 1, fin);
        assert(ret == 1);

        forest->threshes[i] = minv + value * step;
        load(fin, forest->depth, forest->trees[i]);
    }

    read_offsets(fin, forest->offsets[0], forest->rsdlDim, forest->treeSize * leafSize);

    ret = fread(forest->rsdlMean, sizeof(float), ptsSize2, fin); assert(ret == ptsSize2);
    ret = fread(forest->rsdlCoeff, sizeof(float), forest->rsdlDim * ptsSize2, fin); assert(ret == forest->rsdlDim * ptsSize2);
    ret = fread(forest->bias, sizeof(float), forest->rsdlDim, fin); assert(ret == forest->rsdlDim);
}


void release_data(Forest *forest){
    if(forest == NULL)
        return ;

    if(forest->trees != NULL){
        for(int i = 0; i < forest->treeSize; i++)
            release(forest->trees + i);

        delete [] forest->trees;
    }

    forest->trees = NULL;

    if(forest->threshes != NULL)
        delete [] forest->threshes;

    forest->threshes = NULL;

    if(forest->offsets != NULL){
        if(forest->offsets[0] != NULL)
            delete [] forest->offsets[0];
        forest->offsets[0] = NULL;

        delete [] forest->offsets;
    }

    forest->offsets = NULL;

    if(forest->rsdlMean != NULL)
        delete [] forest->rsdlMean;

    forest->rsdlMean = NULL;

    if(forest->rsdlCoeff != NULL)
        delete [] forest->rsdlCoeff;
    forest->rsdlCoeff = NULL;

    forest->capacity = 0;
    forest->treeSize = 0;
}


void release(Forest **forest){
    if(*forest != NULL){
        release_data(*forest);
        delete *forest;
    }

    *forest = NULL;
}


void release_data(NegGenerator *generator){
    if(generator == NULL)
        return ;

    if(generator->fin != NULL)
        fclose(generator->fin);

    generator->fin = NULL;

    generator->forests = NULL;
    generator->fsize = 0;
}


void release(NegGenerator **generator){
    if(*generator != NULL){
        release_data(*generator);
        delete *generator;
    }

    *generator = NULL;
}
