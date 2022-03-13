package com.visionenergy.mycamera;

import android.graphics.Bitmap;
import android.content.Context;

/************************************************
 * 在开始检测人脸之前要先加载模型:loadTracker
 * 检测图像要使用yuv数据，或者灰度图像: trackFace
 * 不再检测人脸，需释放申请的内存:releaseTracker
 ************************************************/

public class FaceUtil {

    public final static int PTS_SIZE = 101;
    public final static int MAX_FACE_SIZE = 3;

    static {
        System.loadLibrary("veFaceSDK");
    };

    public native static int initialSDK(Context context);
    /**********************************************
     * 加载模型
     * detectModel: ./face_lib/detect_model.dat
     * alignModel: ./face_lib/align_model.dat
     * trackModel: ./face_lib/track_model.dat
     **********************************************/
    public native static int loadTracker(String detectModel, String alignModel, String trackModel);


    /*********************************************
     * 追踪人脸
     * imgData:yuv 数据
     * w: 图像宽度
     * h: 图像高度
     * pts: 结果数据[x0, y0, ... , xn, yn]，预先申请空间 > PTS_SIZE * 2 * 3
     * direct: 相机方向
     *********************************************/
    public native static int trackFace(byte[]imgData, int w, int h, float [] pts, int direct);


    /*********************************************
     * 对齐人脸
     * imgData:yuv 数据
     * w: 图像宽度
     * h: 图像高度
     * pts: 结果数据[x0, y0, ... , xn, yn]
     * format: 图像数据格式， 0:NV21, 1:BGRA
     * maxFaceSize: 最多人脸数量，指明pts能保存的人脸数量
     *********************************************/
    public native static int alignFace(byte[] imgData, int w, int h, float [] pts, int format, int maxFaceSize);


    /*********************************************
     * 对齐人脸
     * bitmap: 输入图像
     * pts: 结果数据[x0, y0, ... , xn, yn]
     * maxFaceSize: 最多人脸数量，指明pts能保存的人脸数量
     *********************************************/
    public native static int alignFaceBitmap(Bitmap bitmap, float [] pts, int maxFaceSize);


    /*********************************************
     * 获取欧拉角
     * kpts:人脸特征点
     * faceSize: 人脸数量
     * angles: 欧拉角
     *********************************************/
    public native static void getAngles(float [] kpts, int faceSize, float []angles);


    /********************************************
     * 释放申请的资源
     ********************************************/
    public native static int releaseTracker();
}
