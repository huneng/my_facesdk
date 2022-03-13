#include "tool.h"



#define FIX_INTER_POINT 14

void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts){
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


void affine_image(uint8_t *src, int srcw, int srch, int srcs, HPoint2f cenS,
        uint8_t *dst, int dstw, int dsth, int dsts, HPoint2f cenD, float scale, float angle){

    int FIX_ONE = 1 << FIX_INTER_POINT;
    int FIX_0_5 = FIX_ONE >> 1;

    float sina = sin(-angle) / scale;
    float cosa = cos(-angle) / scale;

    int id = 0;

    int *xtable = new int[(dstw << 1) + (dsth << 1)]; assert(xtable != NULL);
    int *ytable = xtable + (dstw << 1);


    int fcx = cenS.x * FIX_ONE;
    int fcy = cenS.y * FIX_ONE;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y = (i - cenD.y);

        ytable[idx]     = y * sina * FIX_ONE + fcx;
        ytable[idx + 1] = y * cosa * FIX_ONE + fcy;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x = (i - cenD.x);

        xtable[idx]     = x * sina * FIX_ONE;
        xtable[idx + 1] = x * cosa * FIX_ONE;
    }

    id = 0;
    for(int y = 0; y < dsth; y++){
        int idx = y << 1;

        int ys = ytable[idx]    ;
        int yc = ytable[idx + 1];

        for(int x = 0; x < dstw; x++){
            idx = x << 1;

            int xs = xtable[idx];
            int xc = xtable[idx + 1];

            int fx =  xc + ys;
            int fy = -xs + yc;

            int x0 = fx >> FIX_INTER_POINT;
            int y0 = fy >> FIX_INTER_POINT;

            int wx = fx - (x0 << FIX_INTER_POINT);
            int wy = fy - (y0 << FIX_INTER_POINT);

            if(x0 < 0 || x0 >= srcw || y0 < 0 || y0 >= srch)
                continue;

            assert(wx <= FIX_ONE && wy <= FIX_ONE);

            uint8_t *ptr1 = src + y0 * srcs + x0;
            uint8_t *ptr2 = ptr1 + srcs;

            uint8_t value0 = ((ptr1[0] << FIX_INTER_POINT) + (ptr1[1] - ptr1[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;
            uint8_t value1 = ((ptr2[0] << FIX_INTER_POINT) + (ptr2[1] - ptr2[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;

            dst[id + x] = ((value0 << FIX_INTER_POINT) + (value1 - value0) * wy + FIX_0_5) >> FIX_INTER_POINT;
        }

        id += dsts;
    }

    delete [] xtable;
}


void extract_area_from_image(uint8_t *img, int width, int height, int stride, uint8_t *patch, HRect &rect){
    int x0 = rect.x;
    int y0 = rect.y;

    int x1 = x0 + rect.width - 1;
    int y1 = y0 + rect.height - 1;

    int faceSize = HU_MAX(rect.width, rect.height);

    int w, h;

    int bl, bt, br, bb;

    memset(patch, 0, sizeof(uint8_t) * faceSize * faceSize);

    bl = 0, bt = 0, br = 0, bb = 0;

    if(x0 < 0) {
        bl = -x0;
        x0 = 0;
    }

    if(y0 < 0){
        bt = -y0;
        y0 = 0;
    }

    if(x1 > width - 1){
        br = x1 - width + 1;
        x1 = width - 1;
    }

    if(y1 > height - 1){
        bb = y1 - height + 1;
        y1 = height - 1;
    }

    w = faceSize - bl - br;
    h = faceSize - bt - bb;

    patch += bt * faceSize + bl;
    img += y0 * stride + x0;

    for(int y = 0; y < h; y++){
        memcpy(patch, img, sizeof(uint8_t) * w);

        patch += faceSize;
        img += stride;
    }

    rect.x = x0 - bl;
    rect.y = y0 - bt;
}


int read_pts_file(const char *filePath, float *shape, int &ptsSize)
{
    FILE *fin = fopen(filePath, "r");

    char line[256];
    char *ret;

    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 0;
    }

    ret = fgets(line, 255, fin);
    ret = fgets(line, 255, fin);

    sscanf(line, "n_points:  %d", &ptsSize);

    assert(ptsSize <= MAX_PTS_SIZE);

    ret = fgets(line, 255, fin);

    for(int i = 0; i < ptsSize; i++){
        if(fgets(line, 255, fin) == NULL) {
            fclose(fin);
            printf("END of FILE: %s\n", filePath);
            return 0;
        }

        int ret = sscanf(line, "%f %f\n", shape + i, shape + i + ptsSize);

        if(ret == 0) break;
    }

    fclose(fin);


    return ptsSize;
}


int write_pts_file(const char *filePath, float *shape, int ptsSize)
{
    FILE *fout = fopen(filePath, "w");

    char line[256];

    if(fout == NULL)
    {
        printf("Can't open file %s\n", filePath);
        return 1;
    }


    fprintf(fout, "version: 1\n");
    fprintf(fout, "n_points:  %d\n", ptsSize);
    fprintf(fout, "{\n");

    for(int i = 0; i < ptsSize; i++)
        fprintf(fout, "%f %f\n", shape[i], shape[i + ptsSize]);

    fprintf(fout, "}");
    fclose(fout);

    return 0;
}



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

    ssina *= num;
    scosa *= num;

    arg.angle = asinf(ssina);
    arg.scale = var2 / var1;

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




#define Q 14

void bgra2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data)
{
    const short SCALE_B = ((short)(0.1140f * (1 << Q)));
    const short SCALE_G = ((short)(0.5870f * (1 << Q)));
    const short SCALE_R = ((short)(0.2989f * (1 << Q)));

    const short DELTA =  ((short)(1 << (Q - 1)));

    uint8_t *ptrRes;
    uint8_t *ptrSrc;

    int x, y;

    for(y = 0; y <= height - 4; y += 4){
        //0
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //1
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //2
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //3
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }

    for(; y < height; y++){
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }
}



void rgba2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data)
{
    const short SCALE_B = ((short)(0.1140f * (1 << Q)));
    const short SCALE_G = ((short)(0.5870f * (1 << Q)));
    const short SCALE_R = ((short)(0.2989f * (1 << Q)));

    const short DELTA =  ((short)(1 << (Q - 1)));

    uint8_t *ptrRes;
    uint8_t *ptrSrc;

    int x, y;

    for(y = 0; y <= height - 4; y += 4){
        //0
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //1
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //2
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //3
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }

    for(; y < height; y++){
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }
}


static void transpose_wx8_c_code_channel_one(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride, int width)
{
    int i;

    for (i = 0; i < width; i++) {
        dst[0] = src[0 * src_stride];
        dst[1] = src[1 * src_stride];
        dst[2] = src[2 * src_stride];
        dst[3] = src[3 * src_stride];
        dst[4] = src[4 * src_stride];
        dst[5] = src[5 * src_stride];
        dst[6] = src[6 * src_stride];
        dst[7] = src[7 * src_stride];

        src++;
        dst += dst_stride;
    }
}


static void transpose_wx8_c_code_channel_three(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride, int width)
{
    int i;
    for (i = 0; i < width; i++) {
        dst[0] = src[0 * src_stride + 0];
        dst[1] = src[0 * src_stride + 1];
        dst[2] = src[0 * src_stride + 2];

        dst[3] = src[1 * src_stride + 0];
        dst[4] = src[1 * src_stride + 1];
        dst[5] = src[1 * src_stride + 2];

        dst[6] = src[2 * src_stride + 0];
        dst[7] = src[2 * src_stride + 1];
        dst[8] = src[2 * src_stride + 2];

        dst[9] = src[3 * src_stride + 0];
        dst[10] = src[3 * src_stride + 1];
        dst[11] = src[3 * src_stride + 2];

        dst[12] = src[4 * src_stride + 0];
        dst[13] = src[4 * src_stride + 1];
        dst[14] = src[4 * src_stride + 2];

        dst[15] = src[5 * src_stride + 0];
        dst[16] = src[5 * src_stride + 1];
        dst[17] = src[5 * src_stride + 2];

        dst[18] = src[6 * src_stride + 0];
        dst[19] = src[6 * src_stride + 1];
        dst[20] = src[6 * src_stride + 2];

        dst[21] = src[7 * src_stride + 0];
        dst[22] = src[7 * src_stride + 1];
        dst[23] = src[7 * src_stride + 2];

        src += 3;
        dst += dst_stride;
    }

}


static void transpose_wxh_c_code(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    int i,j;
    if (channels == 1){
        for (i = 0; i < width; i++) {
            j = 0;
            for (; j <= height - 4; j += 4) {
                dst[i * dst_stride + j + 0] = src[(j + 0) * src_stride + i];
                dst[i * dst_stride + j + 1] = src[(j + 1) * src_stride + i];
                dst[i * dst_stride + j + 2] = src[(j + 2) * src_stride + i];
                dst[i * dst_stride + j + 3] = src[(j + 3) * src_stride + i];
            }

            for (; j < height; ++j) {
                dst[i * dst_stride + j] = src[j * src_stride + i];
            }
        }
    } else if (channels == 3){
        for (i = 0; i < width; i++) {
            j = 0;
            for (; j <= height - 4; j += 4) {
                dst[i * dst_stride + j + 0] = src[(j + 0) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 1] = src[(j + 0) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 2] = src[(j + 0) * src_stride + 3 * i + 2];

                dst[i * dst_stride + j + 3] = src[(j + 1) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 4] = src[(j + 1) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 5] = src[(j + 1) * src_stride + 3 * i + 2];

                dst[i * dst_stride + j + 6] = src[(j + 2) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 7] = src[(j + 2) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 8] = src[(j + 2) * src_stride + 3 * i + 2];

                dst[i * dst_stride + j + 9] = src[(j + 3) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 10] = src[(j + 3) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 11] = src[(j + 3) * src_stride + 3 * i + 2];
            }
            for (; j < height; ++j) {
                dst[i * dst_stride + j + 0] = src[j * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 1] = src[j * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 2] = src[j * src_stride + 3 * i + 2];
            }
        }
    }  else {
    }
}



static void transpose_plane(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    int i = height;

    if (channels == 1){

        while (i >= 8) {
            transpose_wx8_c_code_channel_one(src, src_stride, dst, dst_stride, width);
            src += 8 * src_stride;
            dst += 8;
            i -= 8;
        }

        if (i > 0) {
            transpose_wxh_c_code(src, src_stride, dst, dst_stride, width, i, 1);
        }

    } else if (channels == 3){
        while (i >= 8) {
            transpose_wx8_c_code_channel_three(src, src_stride, dst, dst_stride, width);
            src += 8 * src_stride;
            dst += 8 * 3;
            i -= 8;
        }

        if (i > 0) {
            transpose_wxh_c_code(src, src_stride, dst, dst_stride, width, i, 3);
        }
    }
}



static void rotation_plane_90(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    src       += src_stride * (height - 1);
    src_stride = -src_stride;
    transpose_plane(src, src_stride, dst, dst_stride, width, height, channels);
}


static void rotation_plane_270(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    dst       += dst_stride * (width - 1);
    dst_stride = -dst_stride;
    transpose_plane(src, src_stride, dst, dst_stride, width, height, channels);
}


static void mirror_copy_row_c_code(const uint8_t* src, uint8_t* dst, int width)
{
    int i;
    for (i = 0; i < width; i++){
        dst[i] = src[width - 1 - i];
    }
}


static void rotate_plane_180(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{

    uint8_t* row           = (uint8_t*)malloc(width);
    const uint8_t* src_bot = src + src_stride * (height - 1);
    uint8_t* dst_bot       = dst + dst_stride * (height - 1);
    int half_height     = (height + 1) >> 1;
    int y;
    void(*ptr_mirror_copy_row)(const uint8_t* src, uint8_t* dst, int width) = mirror_copy_row_c_code;

    for (y = 0; y < half_height; ++y) {
        ptr_mirror_copy_row(src, row, width);
        src += src_stride;
        ptr_mirror_copy_row(src_bot, dst, width);
        dst += dst_stride;
        memcpy(dst_bot, row, width);
        src_bot -= src_stride;
        dst_bot -= dst_stride;
    }
    free(row);
}


static void copy_plane(uint8_t* src, int src_stride, uint8_t* dst, int dst_stride, int width, int height)
{
    int y;
    for (y = 0; y < height; y++) {
        memcpy(dst, src, width);
        src += src_stride;
        dst += dst_stride;
    }
}


void rotate_width_degree(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int &dstw, int &dsth, int &dsts, int degree)
{

    if(degree == 0){
        memcpy(dst, src, sizeof(uint8_t) * srcs * srch);
        dstw = srcw;
        dsth = srch;
        dsts = srcs;
    }
    else if(degree == 1){
        dstw = srch;
        dsth = srcw;
        dsts = srch;

        rotation_plane_90(src, srcs, dst, dsts, srcw, srch, 1);
    }
    else if(degree == 2){
        dstw = srcw;
        dsth = srch;
        dsts = srcs;
        rotate_plane_180(src, srcs, dst, dsts, srcw, srch, 1);
    }
    else if(degree == 3){
        dstw = srch;
        dsth = srcw;
        dsts = srch;

        rotation_plane_270(src, srcs, dst, dsts, srcw, srch, 1);
    }
}


void* aligned_malloc(size_t len, size_t align)
{
    len = len + align + 1;

    void *mem = malloc(len); assert(mem != NULL);

    if(align == 0 || align == 1){
        ((uint8_t*)mem)[0] = 1;
        return (void*)((size_t)mem + 1);
    }

    size_t offset = align - (size_t(mem)) % align;

    size_t ptr = size_t(mem) + offset;
    ((uint8_t*)ptr)[-1] = offset;

    return (void*)ptr;
}


void aligned_free(void *ptr)
{
    size_t addr = size_t(ptr) - ((uint8_t*)ptr)[-1];

    free((void*)addr);
}

