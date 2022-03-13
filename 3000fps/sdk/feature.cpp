#include "feature.h"


void similarity_transform(Shape &src, Shape &dst, TranArgs &arg){
    int ptsSize = src.ptsSize;
    float cx1 = 0.0f, cy1 = 0.0f;
    float cx2 = 0.0f, cy2 = 0.0f;

    int i, j;

    for(i = 0, j = ptsSize; i <= ptsSize - 4; i += 4, j += 4){
        cx1 += src.pts[i];
        cx1 += src.pts[i + 1];
        cx1 += src.pts[i + 2];
        cx1 += src.pts[i + 3];

        cy1 += src.pts[j];
        cy1 += src.pts[j + 1];
        cy1 += src.pts[j + 2];
        cy1 += src.pts[j + 3];
    }

    for(; i < ptsSize; i++, j++){
        cx1 += src.pts[i];
        cy1 += src.pts[j];
    }

    cx1 /= ptsSize;
    cy1 /= ptsSize;

    for(i = 0, j = ptsSize; i <= ptsSize - 4; i += 4, j += 4){
        cx2 += dst.pts[i];
        cx2 += dst.pts[i + 1];
        cx2 += dst.pts[i + 2];
        cx2 += dst.pts[i + 3];

        cy2 += dst.pts[j];
        cy2 += dst.pts[j + 1];
        cy2 += dst.pts[j + 2];
        cy2 += dst.pts[j + 3];
    }

    for(; i < ptsSize; i++, j++){
        cx2 += dst.pts[i];
        cy2 += dst.pts[j];
    }

    cx2 /= ptsSize;
    cy2 /= ptsSize;

    float ssina = 0.0f, scosa = 0.0f, num = 0.0f;
    float var1 = 0.0f, var2 = 0.0f;

    for(int i = 0, j = ptsSize; i < ptsSize; i++, j++){
        float sx = src.pts[i] - cx1;
        float sy = src.pts[j] - cy1;

        float dx = dst.pts[i] - cx2;
        float dy = dst.pts[j] - cy2;

        var1 += sqrtf(sx * sx + sy * sy);
        var2 += sqrtf(dx * dx + dy * dy);

        ssina += (sy * dx - sx * dy);
        scosa += (sx * dx + sy * dy);
    }

    num = 1.0f / sqrtf(ssina * ssina + scosa * scosa);

    ssina = ssina * num;
    scosa = scosa * num;

    arg.angle = asinf(ssina);
    arg.scale = var2 / var1;

    arg.ssina = arg.scale * ssina;
    arg.scosa = arg.scale * scosa;

    arg.cen1.x = cx1;
    arg.cen1.y = cy1;
    arg.cen2.x = cx2;
    arg.cen2.y = cy2;
}


void affine_shape(Shape &shapeSrc, Shape &shapeDst, TranArgs &arg){
    int ptsSize = shapeSrc.ptsSize;

    float sina = sin(arg.angle) * arg.scale;
    float cosa = cos(arg.angle) * arg.scale;

    float *ptrSx = shapeSrc.pts;
    float *ptrSy = shapeSrc.pts + ptsSize;

    float *ptrDx = shapeDst.pts;
    float *ptrDy = shapeDst.pts + ptsSize;

    for(int i = 0; i < ptsSize; i++){
        float x = ptrSx[i] - arg.cen1.x;
        float y = ptrSy[i] - arg.cen1.y;

        ptrDx[i] =  x * cosa + y * sina + arg.cen2.x;
        ptrDy[i] = -x * sina + y * cosa + arg.cen2.y;
    }

    shapeDst.ptsSize = ptsSize;
}

#define FIX_INTER_POINT 14

void affine_sample(uint8_t *img, int width, int height, int stride, Shape &shape, TranArgs &arg)
{
    uint8_t *imgBuffer = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    memset(imgBuffer, 0, sizeof(uint8_t) * width * height);

    int FIX_ONE = 1 << FIX_INTER_POINT;
    int FIX_0_5 = FIX_ONE >> 1;

    int dstw = width;
    int dsth = height;
    int dsts = width;

    uint8_t *dst = imgBuffer;

    float sina = sinf(-arg.angle) / arg.scale;
    float cosa = cosf(-arg.angle) / arg.scale;

    int id = 0;

    int *xtable = new int[(dstw << 1) + (dsth << 1)]; assert(xtable != NULL);
    int *ytable = xtable + (dstw << 1);


    int fcx = arg.cen1.x * FIX_ONE;
    int fcy = arg.cen1.y * FIX_ONE;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y = (i - arg.cen2.y);

        ytable[idx]     = y * sina * FIX_ONE + fcx;
        ytable[idx + 1] = y * cosa * FIX_ONE + fcy;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x = (i - arg.cen2.x);

        xtable[idx]     = x * sina * FIX_ONE;
        xtable[idx + 1] = x * cosa * FIX_ONE;
    }

    affine_shape(shape, shape, arg);


    id = 0;
    for(int y = 0; y < dsth; y++){
        int idy = y << 1;

        int ys = ytable[idy]    ;
        int yc = ytable[idy + 1];

        for(int x = 0; x < dstw; x++){
            int idx = x << 1;

            int xs = xtable[idx];
            int xc = xtable[idx + 1];

            int fx =  xc + ys;
            int fy = -xs + yc;

            int x0 = fx >> FIX_INTER_POINT;
            int y0 = fy >> FIX_INTER_POINT;

            int wx = fx - (x0 << FIX_INTER_POINT);
            int wy = fy - (y0 << FIX_INTER_POINT);

            if(x0 < 0 || x0 >= width || y0 < 0 || y0 >= height)
                continue;


            uint8_t *ptr1 = img + y0 * stride + x0;
            uint8_t *ptr2 = ptr1 + stride;

            uint8_t value0 = ((ptr1[0] << FIX_INTER_POINT) + (ptr1[1] - ptr1[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;
            uint8_t value1 = ((ptr2[0] << FIX_INTER_POINT) + (ptr2[1] - ptr2[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;

            dst[id + x] = ((value0 << FIX_INTER_POINT) + (value1 - value0) * wy + FIX_0_5) >> FIX_INTER_POINT;
        }

        id += dsts;
    }

    delete [] xtable;

    id = 0;

    for(int y = 0; y < height; y++, id += width, img += stride)
        memcpy(img, imgBuffer + id, sizeof(uint8_t) * width);

    delete [] imgBuffer;

}


HRect get_shape_rect(Shape &shape){
    float minx = shape.pts[0], maxx = minx;
    float miny = shape.pts[shape.ptsSize], maxy = miny;

    HRect rect;

    for(int i = 1; i < shape.ptsSize; i++){
        float x = shape.pts[i];
        float y = shape.pts[i + shape.ptsSize];

        minx = HU_MIN(minx, x);
        maxx = HU_MAX(maxx, x);
        miny = HU_MIN(miny, y);
        maxy = HU_MAX(maxy, y);
    }

    float cx = 0.5f * (minx + maxx);
    float cy = 0.5f * (miny + maxy);

    float faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1);

    rect.x = cx - faceSize * 0.5f;
    rect.y = cy - faceSize * 0.5f;
    rect.width = rect.height = faceSize + 0.5f;

    return rect;
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


uint8_t predict(JTree *root, uint8_t *img, int stride, Shape &shape){
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);
    root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    while(root->left != NULL)
        root = root->left + (diff_feature(img, stride, shape, root->featType) > root->thresh);

    return root->leafID;
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


void release(JTree **root){
    if(*root != NULL){
        delete [] *root;
    }

    *root = NULL;
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

