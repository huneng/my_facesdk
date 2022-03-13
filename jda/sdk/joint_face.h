#ifndef _JOINT_FACE_H_
#define _JOINT_FACE_H_


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <memory.h>

#define HU_SWAP(x, y, type) {type tmp = (x); (x) = (y); (y) = (tmp);}
#define HU_MIN(i, j) ((i) > (j) ? (j) : (i))
#define HU_MAX(i, j) ((i) < (j) ? (j) : (i))

#define EPSILON 0.000001f
#define HU_PI 3.1415926535

//quick sort implementation, fast than qsort in stdlib.h
#define IMPLEMENT_QSORT(function_name, T, LT)                                   \
    void function_name( T *array, int total_num)                                        \
{                                                                                   \
    int isort_thresh = 7;                                                           \
    int sp = 0;                                                                     \
    \
    struct                                                                          \
    {                                                                               \
        T *lb;                                                                      \
        T *ub;                                                                      \
    }                                                                               \
    stack[48];                                                                      \
    \
    if( total_num <= 1 )                                                            \
    return;                                                                     \
    \
    stack[0].lb = array;                                                            \
    stack[0].ub = array + (total_num - 1);                                          \
    \
    while( sp >= 0 )                                                                \
    {                                                                               \
        T* left = stack[sp].lb;                                                     \
        T* right = stack[sp--].ub;                                                  \
        \
        for(;;)                                                                     \
        {                                                                           \
            int i, n = (int)(right - left) + 1, m;                                  \
            T* ptr;                                                                 \
            T* ptr2;                                                                \
            \
            if( n <= isort_thresh )                                                 \
            {                                                                       \
                insert_sort_##func_name:                                                \
                for( ptr = left + 1; ptr <= right; ptr++ )                          \
                {                                                                   \
                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1]); ptr2--)   \
                    HU_SWAP( ptr2[0], ptr2[-1], T);                            \
                }                                                                   \
                break;                                                              \
            }                                                                       \
            else                                                                    \
            {                                                                       \
                T* left0;                                                           \
                T* left1;                                                           \
                T* right0;                                                          \
                T* right1;                                                          \
                T* pivot;                                                           \
                T* a;                                                               \
                T* b;                                                               \
                T* c;                                                               \
                int swap_cnt = 0;                                                   \
                \
                left0 = left;                                                       \
                right0 = right;                                                     \
                pivot = left + (n/2);                                               \
                \
                if( n > 40 )                                                        \
                {                                                                   \
                    int d = n / 8;                                                  \
                    a = left, b = left + d, c = left + 2*d;                         \
                    left = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))     \
                    : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                    \
                    a = pivot - d, b = pivot, c = pivot + d;                        \
                    pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
                    : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                    \
                    a = right - 2*d, b = right - d, c = right;                      \
                    right = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
                    : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                }                                                                   \
                \
                a = left, b = pivot, c = right;                                     \
                pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))        \
                : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));       \
                if( pivot != left0 )                                                \
                {                                                                   \
                    HU_SWAP( *pivot, *left0, T );                                  \
                    pivot = left0;                                                  \
                }                                                                   \
                left = left1 = left0 + 1;                                           \
                right = right1 = right0;                                            \
                \
                for(;;)                                                             \
                {                                                                   \
                    while( left <= right && !LT(*pivot, *left) )                    \
                    {                                                               \
                        if( !LT(*left, *pivot) )                                    \
                        {                                                           \
                            if( left > left1 )                                      \
                            HU_SWAP( *left1, *left, T );                       \
                            swap_cnt = 1;                                           \
                            left1++;                                                \
                        }                                                           \
                        left++;                                                     \
                    }                                                               \
                    \
                    while( left <= right && !LT(*right, *pivot) )                   \
                    {                                                               \
                        if( !LT(*pivot, *right) )                                   \
                        {                                                           \
                            if( right < right1 )                                    \
                            HU_SWAP( *right1, *right, T);                      \
                            swap_cnt = 1;                                           \
                            right1--;                                               \
                        }                                                           \
                        right--;                                                    \
                    }                                                               \
                    \
                    if( left > right )                                              \
                    break;                                                      \
                    HU_SWAP( *left, *right, T );                                   \
                    swap_cnt = 1;                                                   \
                    left++;                                                         \
                    right--;                                                        \
                }                                                                   \
                \
                if( swap_cnt == 0 )                                                 \
                {                                                                   \
                    left = left0, right = right0;                                   \
                    goto insert_sort_##func_name;                                   \
                }                                                                   \
                \
                n = HU_MIN( (int)(left1 - left0), (int)(left - left1) );           \
                for( i = 0; i < n; i++ )                                            \
                HU_SWAP( left0[i], left[i-n], T );                             \
                \
                n = HU_MIN( (int)(right0 - right1), (int)(right1 - right) );       \
                for( i = 0; i < n; i++ )                                            \
                HU_SWAP( left[i], right0[i-n+1], T );                          \
                n = (int)(left - left1);                                            \
                m = (int)(right1 - right);                                          \
                if( n > 1 )                                                         \
                {                                                                   \
                    if( m > 1 )                                                     \
                    {                                                               \
                        if( n > m )                                                 \
                        {                                                           \
                            stack[++sp].lb = left0;                                 \
                            stack[sp].ub = left0 + n - 1;                           \
                            left = right0 - m + 1, right = right0;                  \
                        }                                                           \
                        else                                                        \
                        {                                                           \
                            stack[++sp].lb = right0 - m + 1;                        \
                            stack[sp].ub = right0;                                  \
                            left = left0, right = left0 + n - 1;                    \
                        }                                                           \
                    }                                                               \
                    else                                                            \
                    left = left0, right = left0 + n - 1;                        \
                }                                                                   \
                else if( m > 1 )                                                    \
                left = right0 - m + 1, right = right0;                          \
                else                                                                \
                break;                                                          \
            }                                                                       \
        }                                                                           \
    }                                                                               \
}


#define REPEAT_LINE_2(line) \
    line; \
    line;

#define REPEAT_LINE_4(line) \
    REPEAT_LINE_2(line) \
    REPEAT_LINE_2(line)

#define REPEAT_LINE_8(line) \
    REPEAT_LINE_4(line) \
    REPEAT_LINE_4(line)

#define REPEAT_LINE_16(line) \
    REPEAT_LINE_8(line) \
    REPEAT_LINE_8(line)

#define REPEAT_LINE_32(line) \
    REPEAT_LINE_16(line) \
    REPEAT_LINE_16(line)

#define REPEAT_LINE_64(line) \
    REPEAT_LINE_32(line) \
    REPEAT_LINE_32(line)

#define REPEAT_LINE_128(line) \
    REPEAT_LINE_64(line) \
    REPEAT_LINE_64(line)

#define REPEAT_LINE_256(line) \
    REPEAT_LINE_128(line) \
    REPEAT_LINE_128(line)


typedef struct{
    int x;
    int y;
    int width;
    int height;
}HRect;


#define JDA_PTS_SIZE 68
#define MAX_JDA_PTS_SIZE 68

#define OBJECT_FACTOR 1.2f
typedef struct {
    float x;
    float y;
} HPoint2f;

#define TRANS_Q 14

typedef struct{
    float sina;
    float cosa;
    float scale;

    HPoint2f cen1, cen2;

    float ssina;
    float scosa;
} JTranArgs;


typedef struct {
    HPoint2f pts[MAX_JDA_PTS_SIZE];
    int ptsSize;
} Shape;


typedef struct FeatType_t
{
    uint8_t pntIdx1, pntIdx2;
    float off1X, off1Y;
    float off2X, off2Y;
} FeatType;


typedef struct Node_t
{
    FeatType featType;
    int16_t thresh;

    struct Node_t *left;
    struct Node_t *right;

    uint8_t leafID;
    float score;

    //for debug
    int posSize, negSize;
    double pw, nw;
    uint8_t flag;
} Node;


typedef Node Tree;


typedef struct {
    Tree **trees;

    float *threshes;

    int capacity;
    int treeSize;
    int depth;

    float **offsets;

    float *rsdlMean;
    float *rsdlCoeff;

    int rsdlDim;

    float bias[MAX_JDA_PTS_SIZE * 2];
} Forest;


typedef struct {
    Shape meanShape;

    int WINW;
    int WINH;

    Forest *forests;
    int ssize;
    int capacity;

    //detect factor
    float sImgScale;
    float eImgScale;
    float sOffScale;
    float eOffScale;

    int layer;
} JDADetector;


int load(const char *filePath, JDADetector **detector);

void init_detect_factor(JDADetector *detector, float sImgScale, float eImgScale, float sOffScale, float eOffScale, int layer);

int predict(JDADetector *detector, uint8_t *img, int width, int height, int stride);

int detect(JDADetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, Shape **resShapes, float **resScores);
int detect(JDADetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, float **resScores);

void release(JDADetector **detector);


#endif
