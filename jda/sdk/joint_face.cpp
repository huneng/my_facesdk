#include "joint_face.h"


#define FIX_INTER_POINT 14

static void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts){
    uint16_t *table = NULL;

    uint16_t FIX_0_5 = 1 << (FIX_INTER_POINT - 1);
    float scalex, scaley;

    scalex = srcw / float(dstw);
    scaley = srch / float(dsth);

    table = new uint16_t[dstw * 3];

    for(int i = 0; i < dstw; i++){
        float x = i * scalex;

        if(x < 0) x = 0;
        if(x >= srcw - 1) x = srcw - 1.01f;

        int x0 = int(x);

        table[i * 3] = x0;
        table[i * 3 + 2] = (x - x0) * (1 << FIX_INTER_POINT);
        table[i * 3 + 1] = (1 << FIX_INTER_POINT) - table[i * 3 + 2];
    }

    int sId = 0, dId = 0;

    for(int y = 0; y < dsth; y++){
        int x;
        float yc;

        uint16_t wy0, wy1;
        uint16_t y0, y1;
        uint16_t *ptrTab = table;
        int buffer[8];

        yc = y * scaley;

        if(yc < 0) yc = 0;
        if(yc >= srch - 1) yc = srch - 1.01f;

        y0 = uint16_t(yc);
        y1 = y0 + 1;

        wy1 = uint16_t((yc - y0) * (1 << FIX_INTER_POINT));
        wy0 = (1 << FIX_INTER_POINT) - wy1;

        sId = y0 * srcs;

        uint8_t *ptrDst = dst + dId;

        for(x = 0; x <= dstw - 4; x += 4){
            uint16_t x0, x1, wx0, wx1;
            uint8_t *ptrSrc0, *ptrSrc1;

            //1
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[0] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[1] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //2
            x0 = ptrTab[3], x1 = x0 + 1;

            wx0 = ptrTab[4], wx1 = ptrTab[5];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[2] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[3] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //3
            x0 = ptrTab[6], x1 = x0 + 1;

            wx0 = ptrTab[7], wx1 = ptrTab[8];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[4] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[5] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //4
            x0 = ptrTab[9], x1 = x0 + 1;
            wx0 = ptrTab[10], wx1 = ptrTab[11];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[6] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[7] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst[0] = (wy0 * (buffer[0] - buffer[1]) + (buffer[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[1] = (wy0 * (buffer[2] - buffer[3]) + (buffer[3] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[2] = (wy0 * (buffer[4] - buffer[5]) + (buffer[5] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[3] = (wy0 * (buffer[6] - buffer[7]) + (buffer[7] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst += 4;
            ptrTab += 12;
        }

        for(; x < dstw; x++){
            uint16_t x0, x1, wx0, wx1, valuex0, valuex1;

            uint8_t *ptrSrc0, *ptrSrc1;
            x0 = ptrTab[0], x1 = x0 + 1;

            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            valuex0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            valuex1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            dst[y * dsts + x] = (wy0 * (valuex0 - valuex1) + (valuex1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrTab += 3;
        }

        dId += dsts;
    }


    delete [] table;
}


static uint8_t* mean_filter_3x3_res(uint8_t *img, int width, int height, int stride){

    uint8_t *buffer = new uint8_t[width * height];

    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){
            uint8_t *ptr = img + y * stride + x;

            uint32_t value = ptr[-stride - 1] + ptr[-stride] + ptr[-stride + 1] +
                             ptr[-1] + ptr[0] + ptr[1] +
                             ptr[ stride - 1] + ptr[ stride] + ptr[stride + 1];

            buffer[y * width + x] = value * 0.111111f;
        }
    }

    return buffer;
}


static void similarity_transform(Shape &src, Shape &dst, JTranArgs &arg){
    int ptsSize = src.ptsSize;
    float cx1 = 0.0f, cy1 = 0.0f;
    float cx2 = 0.0f, cy2 = 0.0f;

    int i;

    for(i = 0; i <= ptsSize - 4; i += 4){
        cx1 += src.pts[i].x;
        cy1 += src.pts[i].y;
        cx1 += src.pts[i + 1].x;
        cy1 += src.pts[i + 1].y;
        cx1 += src.pts[i + 2].x;
        cy1 += src.pts[i + 2].y;
        cx1 += src.pts[i + 3].x;
        cy1 += src.pts[i + 3].y;
    }

    for(; i < ptsSize; i++){
        cx1 += src.pts[i].x;
        cy1 += src.pts[i].y;
    }

    cx1 /= ptsSize;
    cy1 /= ptsSize;

    for(i = 0; i <= ptsSize - 4; i += 4){
        cx2 += dst.pts[i].x;
        cy2 += dst.pts[i].y;
        cx2 += dst.pts[i + 1].x;
        cy2 += dst.pts[i + 1].y;
        cx2 += dst.pts[i + 2].x;
        cy2 += dst.pts[i + 2].y;
        cx2 += dst.pts[i + 3].x;
        cy2 += dst.pts[i + 3].y;
    }

    for(; i < ptsSize; i++){
        cx2 += dst.pts[i].x;
        cy2 += dst.pts[i].y;
    }

    cx2 /= ptsSize;
    cy2 /= ptsSize;

    float ssina = 0.0f, scosa = 0.0f, num = 0.0f;
    float var1 = 0.0f, var2 = 0.0f;

    for(int i = 0; i < ptsSize; i++){
        float sx = src.pts[i].x - cx1;
        float sy = src.pts[i].y - cy1;

        float dx = dst.pts[i].x - cx2;
        float dy = dst.pts[i].y - cy2;

        var1 += sqrtf(sx * sx + sy * sy);
        var2 += sqrtf(dx * dx + dy * dy);

        ssina += (sy * dx - sx * dy);
        scosa += (sx * dx + sy * dy);
    }

    num = 1.0f / sqrtf(ssina * ssina + scosa * scosa);

    arg.sina = ssina * num;
    arg.cosa = scosa * num;

    arg.scale = var2 / var1;

    arg.cen1.x = cx1;
    arg.cen1.y = cy1;
    arg.cen2.x = cx2;
    arg.cen2.y = cy2;

    arg.ssina = arg.scale * arg.sina;
    arg.scosa = arg.scale * arg.cosa;
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
        FeatType &featType, JTranArgs &arg)
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


static float predict(Tree *root, uint8_t *img, int stride, Shape &shape, uint8_t &leafID){
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    while(root->left != NULL)
        root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    leafID = root->leafID;

    return root->score;
}


static float predict(Tree *root, uint8_t *img, int stride, Shape &curShape, JTranArgs &arg, uint8_t &leafID){
    root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);
    root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);
    root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);

    while(root->left != NULL)
        root = root->left + (diff_feature(img, stride, curShape, root->featType, arg) > root->thresh);

    leafID = root->leafID;

    return root->score;
}


static void load(FILE *fin, int depth, Tree *root){
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


static void release(Tree **root){
    if(*root != NULL)
        delete [] *root;

    *root = NULL;
}



static int predict_one(Forest *forest, Shape &meanShape, uint8_t *img, int stride, Shape &shape, float &score){
    score = 0;
    if(forest->treeSize == 0)
        return 1;

    uint8_t leafIDs[512];
    int ptsSize = shape.ptsSize;
    int ptsSize2 = shape.ptsSize << 1;

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


    float buffer[MAX_JDA_PTS_SIZE * 2];
    float rsdl[MAX_JDA_PTS_SIZE * 2];

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


static int predict(Forest *forest, Shape &meanShape, uint8_t *img, int stride, Shape &curShape, float &score){
    score = 0;

    if(forest->treeSize == 0)
        return 1;

    JTranArgs arg;
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

    float buffer[MAX_JDA_PTS_SIZE * 2];
    float rsdl[MAX_JDA_PTS_SIZE * 2];

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


static int predict(Forest *forest, int size, Shape &meanShape, uint8_t *img, int stride, Shape &shape, float &score){
    for(int i = 0; i < size; i++){
        if(predict(forest + i, meanShape, img, stride, shape, score) == 0)
            return 0;
    }

    return 1;
}


static int validate(Forest *forest, int tsize, Shape &meanShape, uint8_t *img, int stride, Shape &shape){
    JTranArgs arg;
    float score = 0;
    uint8_t leafID;

    tsize = HU_MIN(tsize, forest->treeSize);

    similarity_transform(meanShape, shape, arg);

    for(int i = 0; i < tsize; i++){
        score += predict(forest->trees[i], img, stride, shape, arg, leafID);
        if(score <= forest->threshes[i])
            return 0;
    }

    return 1;
}


static void read_offsets(FILE *fin, float *offsets, int ptsSize2, int allLeafSize){
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


static void load(FILE *fin, int ptsSize, Forest *forest){
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


static void release_data(Forest *forest){
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


static void release(Forest **forest){
    if(*forest != NULL){
        release_data(*forest);
        delete *forest;
    }

    *forest = NULL;
}


static int predict(JDADetector *detector, uint8_t *img, int stride, Shape &shape, float &score){
#if 1
    if(predict_one(detector->forests, detector->meanShape, img, stride, shape, score) == 0)
        return 0;

    for(int i = 1; i < detector->ssize; i++)
        if(predict(detector->forests + i, detector->meanShape, img, stride, shape, score) == 0)
            return 0;
#else
    for(int i = 0; i < detector->ssize; i++)
        if(predict(detector->forests + i, detector->meanShape, img, stride, shape, score) == 0)
            return 0;

#endif
    return 1;
}


int predict(JDADetector *detector, uint8_t *img, int width, int height, int stride){
    uint8_t *sImg;
    int WINW = detector->WINW;
    int WINH = detector->WINH;
    int WINS = stride;
    int ret = 0;
    Shape shape;

    if(width == WINW && height == WINH)
        sImg = img;
    else {
        sImg = new uint8_t[WINW * WINH];
        WINS = WINW;

        resizer_bilinear_gray(img, width, height, stride, sImg, WINW, WINH, WINS);
    }

    shape = detector->meanShape;

    ret = validate(detector->forests, 128, detector->meanShape, img, WINS, shape);

    if(sImg != img)
        delete [] sImg;

    return ret;
}


int load(const char *filePath, JDADetector **rdetector){
    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;
    int ptsSize;

    JDADetector *detector = new JDADetector;

    ret = fread(&detector->meanShape, sizeof(Shape), 1, fin); assert(ret == 1);
    ret = fread(&detector->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&detector->WINH, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&detector->ssize, sizeof(int), 1, fin); assert(ret == 1);

    detector->forests = new Forest[detector->ssize];
    detector->capacity = detector->ssize;

    memset(detector->forests, 0, sizeof(Forest) *detector->ssize);

    ptsSize = detector->meanShape.ptsSize;

    assert(ptsSize == JDA_PTS_SIZE);

    for(int i = 0; i < detector->ssize; i++)
        load(fin, ptsSize, detector->forests + i);

    fclose(fin);

    init_detect_factor(detector, 0.2, 0.9, 0.2, 0.1, 10);

    *rdetector = detector;

    return 0;
}


static int calc_overlapping_area(HRect &rect1, HRect &rect2){
    int cx1 = rect1.x + rect1.width / 2;
    int cy1 = rect1.y + rect1.height / 2;
    int cx2 = rect2.x + rect2.width / 2;
    int cy2 = rect2.y + rect2.height / 2;

    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;

    if(abs(cx1 - cx2) < rect1.width / 2 + rect2.width/2 && abs(cy1 - cy2) < rect1.height / 2 + rect2.height / 2){
        x0 = HU_MAX(rect1.x , rect2.x);
        x1 = HU_MIN(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
        y0 = HU_MAX(rect1.y, rect2.y);
        y1 = HU_MIN(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
    }
    else {
        return 0;
    }

    return (y1 - y0 + 1) * (x1 - x0 + 1);
}


static int merge_rects(HRect *rects, Shape *shapes, float *confs, int size){
    if(size < 2) return size;

    uint8_t *flags = new uint8_t[size];

    memset(flags, 0, sizeof(uint8_t) * size);

    for(int i = 0; i < size; i++){
        if(flags[i] == 1)
            continue;

        float area0 = 1.0f / (rects[i].width * rects[i].height);

        for(int j = i + 1; j < size; j++){
            if(flags[j] == 1) continue;

            float area1 = 1.0f / (rects[j].width * rects[j].height);

            int overlap = calc_overlapping_area(rects[i], rects[j]);

            if(overlap * area1 > 0.6f || overlap * area0 > 0.6f){
                if(confs[i] > confs[j])
                    flags[j] = 1;
                else
                    flags[i] = 1;
            }
        }
    }

    for(int i = 0; i < size; i++){
        if(flags[i] == 0) {
            continue;
        }

        flags[i] = flags[size - 1];

        rects[i] = rects[size - 1];
        confs[i] = confs[size - 1];
        shapes[i] = shapes[size - 1];

        i --;
        size --;
    }

    delete []flags;
    flags = NULL;

    return size;
}


void init_detect_factor(JDADetector *detector, float sImgScale, float eImgScale, float sOffScale, float eOffScale, int layer)
{
    if(sImgScale < eImgScale){
        if(sOffScale < eOffScale)
            HU_SWAP(sOffScale, eOffScale, float);
    }
    else {
        if(sOffScale > eOffScale)
            HU_SWAP(sOffScale, eOffScale, float);
    }

    detector->sImgScale = sImgScale;
    detector->eImgScale = eImgScale;
    detector->sOffScale = sOffScale;
    detector->eOffScale = eOffScale;

    detector->layer = layer;
}


int detect_one_scale(JDADetector *cc, float scale, float offScale,
                     uint8_t *iImg, int width, int height, int stride,
                     HRect *resRect, Shape *resShape, float *resScores){
    int WINW = cc->WINW;
    int WINH = cc->WINH;

    int dx = WINW * offScale;
    int dy = WINH * offScale;

    int count = 0;

    int ptsSize = cc->meanShape.ptsSize;

    for(int y = 0; y <= height - WINH; y += dy){
        for(int x = 0; x <= width - WINW; x += dx){
            uint8_t *ptr = iImg + y * stride + x;

            Shape shape = cc->meanShape;
            float score;

            if(predict(cc, ptr, stride, shape, score) == 1){
                /*
                printf("score: %f\n", score);
                show_shape(ptr, WINW, WINH, stride, shape, "shape");
                //*/
                //
                resRect[count].x = x * scale;
                resRect[count].y = y * scale;

                resRect[count].width  = WINW * scale;
                resRect[count].height = WINH * scale;

                resScores[count] = score;

                for(int p = 0; p < ptsSize; p++){
                    resShape[count].pts[p].x = (shape.pts[p].x + x) * scale;
                    resShape[count].pts[p].y = (shape.pts[p].y + y) * scale;
                }

                resShape[count].ptsSize = ptsSize;

                count++;
            }
        }
    }

    count = merge_rects(resRect, resShape, resScores, count);

    return count;
}


static int calculate_max_size(int width, int height, float scale, int winSize){
    int minwh = HU_MIN(width, height);

    assert(scale < 1.0f);

    int size = minwh * scale;
    float scale2 = (float)winSize / size;

    width ++;
    height ++;

    if(scale2 < 1)
        return width * height;

    return (width * scale2 + 0.5f) * (height * scale2 + 0.5f);
}


static HRect get_face_rect(Shape *shape, int width, int height){
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    for(int i = 0; i < shape->ptsSize; i++){
        minx = HU_MIN(minx, shape->pts[i].x);
        maxx = HU_MAX(maxx, shape->pts[i].x);
        miny = HU_MIN(miny, shape->pts[i].y);
        maxy = HU_MAX(maxy, shape->pts[i].y);
    }

    float cx = (maxx + minx) * 0.5f;
    float cy = (maxy + miny) * 0.5f;

    float faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1) * OBJECT_FACTOR;
    float len = 0;

    HRect rect;

    rect.x = cx - faceSize * 0.5f;
    rect.y = cy - faceSize * 0.5f;
    rect.x = HU_MAX(0, rect.x);
    rect.y = HU_MAX(0, rect.y);

    rect.width  = rect.x + faceSize - 1;
    rect.height = rect.y + faceSize - 1;

    if(rect.width >= width){
        rect.width = faceSize;
        rect.x = width - faceSize;
    }
    else {
        rect.width = rect.width - rect.x + 1;
    }

    if(rect.height > height){
        rect.height = faceSize;
        rect.y = height - faceSize;
    }
    else {
        rect.height = rect.height - rect.y + 1;
    }

    return rect;
}


int detect(JDADetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, Shape **resShapes, float **resScores){
    int WINW, WINH, capacity;
    float imgScale, imgFactor, offScale, offFactor;

    uint8_t *dImg, *ptrSrc, *ptrDst;

    HRect *rects = NULL;
    float *scores = NULL;
    Shape *shapes = NULL;
    int top = 0;

    int srcw, srch, srcs, dstw, dsth, dsts;
    int minSide;
    int count;

    WINW = cc->WINW;
    WINH = cc->WINH;

    imgScale = cc->sImgScale;
    offScale = cc->sOffScale;

    imgFactor = powf(cc->eImgScale / cc->sImgScale, 1.0f / (cc->layer - 1));
    offFactor = powf(cc->eOffScale / cc->sOffScale, 1.0f / (cc->layer - 1));

    capacity = calculate_max_size(width, height, imgScale, HU_MAX(WINW, WINH));

    dImg = new uint8_t[capacity * 2]; assert(dImg != NULL);

    const int BUFFER_SIZE = 500;

    rects  = new HRect[BUFFER_SIZE]; assert(rects != NULL);
    scores = new float[BUFFER_SIZE]; assert(scores != NULL);
    shapes = new Shape[BUFFER_SIZE]; assert(shapes != NULL);

    memset(rects, 0, sizeof(HRect)  * BUFFER_SIZE);
    memset(scores, 0, sizeof(float) * BUFFER_SIZE);
    memset(shapes, 0, sizeof(Shape) * BUFFER_SIZE);

    ptrSrc = img;
    ptrDst = dImg;

    srcw = width;
    srch = height;
    srcs = stride;

    count = 0;

    minSide = HU_MIN(width, height);

    for(int i = 0; i < cc->layer; i++){
        uint8_t *buffer;
        float scale2 = HU_MIN(WINW, WINH) / (minSide * imgScale);

        dstw = scale2 * width;
        dsth = scale2 * height;
        dsts = dstw;

        resizer_bilinear_gray(ptrSrc, srcw, srch, srcs, ptrDst, dstw, dsth, dsts);

        buffer = mean_filter_3x3_res(ptrDst, dstw, dsth, dsts);
        count += detect_one_scale(cc, 1.0f / scale2, offScale, buffer, dstw, dsth, dstw, rects + count, shapes + count, scores + count);
        delete [] buffer;

        ptrSrc = ptrDst;

        srcw = dstw;
        srch = dsth;
        srcs = dsts;

        if(ptrDst == dImg)
            ptrDst = dImg + dstw * dsth;
        else
            ptrDst = dImg;

        imgScale *= imgFactor;
        offScale *= offFactor;
    }

    count = merge_rects(rects, shapes, scores, count);

    if(count > 0){
        //*
        for(int i = 0; i < count; i++)
            rects[i] = get_face_rect(shapes + i, width, height);
        // */

        *resRect = rects;
        *resShapes = shapes;
        *resScores = scores;
    }
    else {
        delete [] rects;
        delete [] shapes;
        delete [] scores;
    }

    delete [] dImg;

    return count;
}


int detect(JDADetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, float **resScores){
    Shape *shapes;
    int rsize;

    rsize = detect(cc, img, width, height, stride, resRect, &shapes, resScores);
    if(rsize > 0)
        delete [] shapes;

    return rsize;
}


static void release_data(JDADetector *detector){
    if(detector != NULL){
        for(int i = 0; i < detector->ssize; i++)
            release_data(detector->forests + i);

        delete [] detector->forests;
    }

    detector->forests = NULL;
    detector->capacity = 0;
    detector->ssize = 0;
}


void release(JDADetector **detector){
    if(*detector != NULL){
        release_data(*detector);
        delete *detector;
    }

    *detector = NULL;
}
