cp -r ./libs/* ~/AndroidStudioProjects/MyCamera/app/src/main/jniLibs/ 
cp -r ./libs/* ./face_lib/

cp ./com_visionenergy_mycamera_FaceUtil.h ./face_lib/ 
cp ./FaceUtil.java ./face_lib/

cp ./face_lib/com_visionenergy_mycamera_FaceUtil.h ~/AndroidStudioProjects/MyCamera/app/src/main/jni/ 
cp ./face_lib/FaceUtil.java ~/AndroidStudioProjects/MyCamera/app/src/main/java/com/visionenergy/mycamera/ 
cp ./face_lib/*.dat ~/AndroidStudioProjects/MyCamera/app/src/main/assets/
