#include <jni.h>
#include <string>

#include <opencv2/opencv.hpp>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_seg_1ml_1cpp_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "jpg from assets -->\nmat -->\nresize -->\ntflite input -->\npredict -->\nparse result -->\nbitmap -->\nimageView";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_seg_1ml_1cpp_MainActivity_capture_video(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMatDst) {

    cv::Mat* matDst = (cv::Mat*) objMatDst;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_seg_1ml_1cpp_MainActivity_processImage(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMatSrc,
        jlong   objMatDst) {

    cv::Mat* matSrc = (cv::Mat*) objMatSrc;
    cv::Mat* matDst = (cv::Mat*) objMatDst;

//    static cv::Mat *matPrevious = NULL;
//    if (matPrevious == NULL) {
//        /* lazy initialization */
//        matPrevious = new cv::Mat(matSrc->rows, matSrc->cols, matSrc->type());
//    }
//    cv::absdiff(*matSrc, *matPrevious, *matDst);
    *matDst = *matSrc;
    return 0;
}