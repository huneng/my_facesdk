#include "com_visionenergy_mycamera_FaceUtil.h"
#include "type_defs.h"
#include "face_manager.h"

#include <android/log.h>
#include <android/bitmap.h>
#include <sys/time.h>


#include "tool.h"

int registerId = -1;
/********************************************
 *校验APP包名和签名是否合法返回值为0表示合法
 ********************************************/
JNIEXPORT jint JNICALL Java_com_visionenergy_mycamera_FaceUtil_initialSDK(JNIEnv*env, jobject thiz, jobject context)
{
//#define MINE_CAMERA
#if defined(MINE_CAMERA)
    const char *app_packageName ="com.visionenergy.mycamera";//此处为app包名
    int app_signature_hash_code = -1328977869;//此处为java层调用getSignature得到的值

#elif defined(XIU_IMAGE)
    const char *app_packageName ="com.imagexiu";//此处为app包名
    int app_signature_hash_code = 185101498;//此处为java层调用getSignature得到的值

#elif defined(FACE_CUTE)
    const char *app_packageName = "com.xiusdk.facecute";
    int app_signature_hash_code = 1962323344;
#else
    const char *app_packageName = "";
    int app_signature_hash_code = 0;
    registerId = 0;
    return 0;
#endif

    jclass context_clazz = env->GetObjectClass(context);

    //得到getPackageManager方法的ID
    jmethodID methodID_getPackageManager=env->GetMethodID(context_clazz,"getPackageManager", "()Landroid/content/pm/PackageManager;");

    //获得PackageManager对象
    jobject packageManager = env->CallObjectMethod(context,
            methodID_getPackageManager);

    //	//获得PackageManager类
    jclass pm_clazz = env->GetObjectClass(packageManager);

    //得到getPackageInfo方法的ID
    jmethodID methodID_pm = env->GetMethodID(pm_clazz,"getPackageInfo","(Ljava/lang/String;I)Landroid/content/pm/PackageInfo;");

    //得到getPackageName方法的ID
    jmethodID methodID_pack = env->GetMethodID(context_clazz,"getPackageName","()Ljava/lang/String;");

    //获得当前应用的包名
    jstring application_package = (jstring)(env->CallObjectMethod(context,methodID_pack));
    const char*package_name = env->GetStringUTFChars(application_package,0);

    //获得PackageInfo
    jobject packageInfo = env->CallObjectMethod(packageManager,methodID_pm,application_package,64);
    jclass packageinfo_clazz = env->GetObjectClass(packageInfo);
    jfieldID fieldID_signatures = env->GetFieldID(packageinfo_clazz,"signatures","[Landroid/content/pm/Signature;");
    jobjectArray signature_arr=(jobjectArray)env->GetObjectField(packageInfo,fieldID_signatures);

    //Signature数组中取出第一个元素
    jobject signature = env->GetObjectArrayElement(signature_arr,0);

    //读signature的hashcode
    jclass signature_clazz = env->GetObjectClass(signature);
    jmethodID methodID_hashcode = env->GetMethodID(signature_clazz,"hashCode","()I");
    jint hashCode = env->CallIntMethod(signature,methodID_hashcode);

    if(strcmp(package_name,app_packageName)!=0 || (hashCode!=app_signature_hash_code)){
        registerId = -1;
        return -1;
    }

    registerId = 0;

    return 0;
}


JNIEXPORT jint JNICALL Java_com_visionenergy_mycamera_FaceUtil_loadTracker
        (JNIEnv *env, jobject, jstring faceDetectModel, jstring faceAlignerModel, jstring faceTrackerModel){
    jboolean copy = false;

    if(registerId != 0) return -1;

    const char *p1 = env->GetStringUTFChars(faceDetectModel, &copy);
    const char *p2 = env->GetStringUTFChars(faceAlignerModel, &copy);
    const char *p3 = env->GetStringUTFChars(faceTrackerModel, &copy);

    __android_log_print(ANDROID_LOG_INFO, "jni", "Load model %s %s %s\n", p1, p2, p3);

    return load_models(p1, p2, p3);
}


#include <sys/time.h>
JNIEXPORT jint JNICALL Java_com_visionenergy_mycamera_FaceUtil_trackFace
        (JNIEnv *env, jobject, jbyteArray yuv, jint w, jint h, jfloatArray pts, int direct){
    if(registerId != 0 )
        return 0;

    jboolean copy = false;

    jbyte  *ptrData = env->GetByteArrayElements(yuv, &copy);
    jfloat *ptrPts  = env->GetFloatArrayElements(pts, &copy);

    int rsize = 0;

    struct timezone tz;
    struct timeval stv, etv;

    gettimeofday(&stv, &tz);
    rsize = process_track((uint8_t*)ptrData, w, h, w, FT_IMAGE_NV21, direct, ptrPts);
    gettimeofday(&etv, &tz);

#ifdef __ARM_NEON
    __android_log_print(ANDROID_LOG_INFO, "jni", "arm neon: %f", 1000.0f * (etv.tv_sec - stv.tv_sec) + 0.001f * (etv.tv_usec - stv.tv_usec));
#else
    __android_log_print(ANDROID_LOG_INFO, "jni", "time: %f",
            1000.0f * (etv.tv_sec - stv.tv_sec) + 0.001f * (etv.tv_usec - stv.tv_usec));

#endif


    env->ReleaseByteArrayElements(yuv, ptrData, 0);
    env->ReleaseFloatArrayElements(pts, ptrPts, JNI_COMMIT);

    return rsize;
}



JNIEXPORT jint JNICALL Java_com_visionenergy_mycamera_FaceUtil_alignFaceBitmap
    (JNIEnv *env, jobject obj, jobject img,  jfloatArray pts, int maxFaceSize){
        if(registerId != 0)
            return -1;

    AndroidBitmapInfo binfo = {};

    int ret = 0;

    if(AndroidBitmap_getInfo(env, img, &binfo) < 0)
        return 0;

    void* ptrData = NULL;

    if (AndroidBitmap_lockPixels(env, img, &ptrData) < 0)
        return 0;

    if(binfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return 0;

    jboolean copy = false;

    jfloat *ptrPts  = env->GetFloatArrayElements(pts, &copy);

    float *kpts;
    int rsize;
    int w = binfo.width;
    int h = binfo.height;

    uint8_t *gray = new uint8_t[w * h];
    rgba2gray((uint8_t *)ptrData, w, h, w * 4, gray);

    rsize = process_align(gray, w, h, w, FT_IMAGE_NV21, &kpts);

    AndroidBitmap_unlockPixels(env, img);

    delete [] gray;

    rsize = HU_MIN(rsize, maxFaceSize);
    if(rsize > 0){
        memcpy(ptrPts, kpts, sizeof(float) * PTS_SIZE2 * rsize);
        delete [] kpts;
    }

    env->ReleaseFloatArrayElements(pts, ptrPts, JNI_COMMIT);

    return rsize;
}


JNIEXPORT jint JNICALL Java_com_visionenergy_mycamera_FaceUtil_alignFace
        (JNIEnv *env, jobject obj, jbyteArray yuv, jint w, jint h, jfloatArray pts, jint format, int maxFaceSize){
            if(registerId != 0)
                return -1;
    jboolean copy = false;

    jbyte  *ptrData = env->GetByteArrayElements(yuv, &copy);
    jfloat *ptrPts  = env->GetFloatArrayElements(pts, &copy);

    float *kpts;
    int ptsSize;

    int rsize;

    rsize = process_align((uint8_t*)ptrData, w, h, w, format, &kpts);
    rsize = HU_MIN(maxFaceSize, rsize);

    if(rsize > 0){
        memcpy(ptrPts, kpts, sizeof(float) * PTS_SIZE2 * rsize);
        delete [] kpts;
    }

    env->ReleaseByteArrayElements(yuv, ptrData, 0);
    env->ReleaseFloatArrayElements(pts, ptrPts, JNI_COMMIT);

    return rsize;
}


JNIEXPORT jint JNICALL Java_com_visionenergy_mycamera_FaceUtil_getAngles
        (JNIEnv *env, jobject obj, jfloatArray kpts, jint faceSize, jfloatArray angles)
{
    jboolean copy = false;

    jfloat *ptrPts = env->GetFloatArrayElements(kpts, &copy);
    jfloat *ptrAngles = env->GetFloatArrayElements(angles, &copy);

    for(int i = 0; i < faceSize; i++){
        int id = i * 3;
        get_angles(ptrPts + i * PTS_SIZE2, PTS_SIZE2, ptrAngles[id + 0], ptrAngles[id + 1], ptrAngles[id + 2]);
    }

    env->ReleaseFloatArrayElements(kpts, ptrPts, 0);
    env->ReleaseFloatArrayElements(angles, ptrAngles, JNI_COMMIT);
}


JNIEXPORT jint JNICALL Java_com_visionenergy_mycamera_FaceUtil_releaseTracker
        (JNIEnv *env, jobject){
    release_models();
}
