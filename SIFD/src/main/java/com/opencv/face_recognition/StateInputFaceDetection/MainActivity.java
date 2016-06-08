package com.opencv.face_recognition.StateInputFaceDetection;

import android.content.Context;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends ActionBarActivity implements CvCameraViewListener {

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeClassifierFace;
    private CascadeClassifier cascadeClassifierEyes;
    private CascadeClassifier cascadeClassifierLEye;
    private CascadeClassifier cascadeClassifierREye;
    private final String TAG = "MAIN";
    private Mat grayscaleImage;
    private int absoluteFaceSize;

    private int rightCounter;
    private int leftCounter;
    private int bothCounter;
    private final int DETECT_THRESHHOLD = 2;



    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:
                    initializeOpenCVDependencies();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    private void initializeOpenCVDependencies(){
        try{
            //the other files for detections (found on: www.embedded-vision.com/forums/2013/01/14/using-face-detection-opencv-java-api)
            /*
            dl: http://alereimondo.no-ip.org/OpenCV/34/

            haarcascade_eye.xml
            haarcascade_mcs_leftear.xml
            haarcascade_eye_tree_eyeglasses.xml
            haarcascade_mcs_lefteye.xml
            haarcascade_frontalface_alt.xml
            haarcascade_mcs_mouth.xml
            haarcascade_frontalface_alt2.xml
            haarcascade_mcs_nose.xml
            haarcascade_frontalface_alt_tree.xml
            haarcascade_mcs_rightear.xml
            haarcascade_frontalface_default.xml
            haarcascade_mcs_righteye.xml
            haarcascade_fullbody.xml
            haarcascade_mcs_upperbody.xml
            haarcascade_lefteye_2splits.xml
            haarcascade_profileface.xml
            haarcascade_lowerbody.xml
            haarcascade_righteye_2splits.xml
            haarcascade_mcs_eyepair_big.xml
            haarcascade_upperbody.xml
            haarcascade_mcs_eyepair_small.xml
            */

            // Copy the resource into a temp file so OpenCV can load it
            InputStream isFace = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDirFace = getDir("cascadeFace", Context.MODE_APPEND);
            File mCascadeFileFace = new File(cascadeDirFace, "lbpcascade_frontalface.xml");



            //save the file for cascade:
            FileOutputStream osFace = new FileOutputStream(mCascadeFileFace);

            byte[] bufferFace = new byte[4096];
            int bytesRead;
            while ((bytesRead = isFace.read(bufferFace)) != -1) {
                osFace.write(bufferFace, 0, bytesRead);
            }
            isFace.close();
            osFace.close();

            // Copy the resource into a temp file so OpenCV can load it
            InputStream isEye = getResources().openRawResource(R.raw.lbpcascade_eyes);
            File cascadeDirEye = getDir("cascadeEye", Context.MODE_APPEND);
            File mCascadeFileEye = new File(cascadeDirEye, "lbpcascade_eyes.xml");

            //save the file for cascade:
            FileOutputStream osEye = new FileOutputStream(mCascadeFileEye);

            byte[] bufferEye = new byte[4096];
            int bytesReadEye;
            while ((bytesReadEye = isEye.read(bufferEye)) != -1) {
                osEye.write(bufferEye, 0, bytesReadEye);
            }
            isEye.close();
            osEye.close();

            // Copy the resource into a temp file so OpenCV can load it
            InputStream isLEye = getResources().openRawResource(R.raw.lbpcascade_leye);
            File cascadeDirLEye = getDir("cascadeLEye", Context.MODE_APPEND);
            File mCascadeFileLEye = new File(cascadeDirLEye, "lbpcascade_leye.xml");

            //save the file for cascade:
            FileOutputStream osLEye = new FileOutputStream(mCascadeFileLEye);

            byte[] bufferLEye = new byte[4096];
            int bytesReadLEye;
            while ((bytesReadLEye = isLEye.read(bufferLEye)) != -1) {
                osLEye.write(bufferLEye, 0, bytesReadLEye);
            }
            isLEye.close();
            osLEye.close();

            // Copy the resource into a temp file so OpenCV can load it
            InputStream isREye = getResources().openRawResource(R.raw.lbpcascade_reye);
            File cascadeDirREye = getDir("cascadeREye", Context.MODE_APPEND);
            File mCascadeFileREye = new File(cascadeDirREye, "lbpcascade_reye.xml");

            //save the file for cascade:
            FileOutputStream osREye = new FileOutputStream(mCascadeFileREye);

            byte[] bufferREye = new byte[4096];
            int bytesReadREye;
            while ((bytesReadREye = isREye.read(bufferREye)) != -1) {
                osREye.write(bufferREye, 0, bytesReadREye);
            }
            isREye.close();
            osREye.close();


            // Load the cascade classifiers
            cascadeClassifierFace = new CascadeClassifier(mCascadeFileFace.getAbsolutePath());
            cascadeClassifierEyes = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
            cascadeClassifierLEye = new CascadeClassifier(mCascadeFileLEye.getAbsolutePath());
            cascadeClassifierREye = new CascadeClassifier(mCascadeFileREye.getAbsolutePath());
            if (cascadeClassifierEyes.empty()|| cascadeClassifierFace.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                cascadeClassifierEyes = null;
                cascadeClassifierFace = null;
                cascadeClassifierLEye = null;
                cascadeClassifierREye = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFileFace.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }

        openCvCameraView.enableView();
        openCvCameraView.enableFpsMeter();  //enable fps meter
        openCvCameraView.setCameraIndex(1); //switch to front cam with 1

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        openCvCameraView = new JavaCameraView(this, -1);
        openCvCameraView.setMaxFrameSize(640,480); //set max resolution
        setContentView(openCvCameraView);
        openCvCameraView.setCvCameraViewListener(this);

        //setContentView(R.layout.activity_main);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {

        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        if (id == R.id.action_settings) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        // The faces will be a 20% of the height of the screen,
        //setup the face detection size
        absoluteFaceSize = (int) (height * 0.2);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        // Create a grayscale image
        Imgproc.cvtColor(aInputFrame, grayscaleImage, Imgproc.COLOR_RGBA2RGB);
        //matrix for rectangle
        MatOfRect faces = new MatOfRect();
        MatOfRect eyes = new MatOfRect();
        MatOfRect leyes = new MatOfRect();
        MatOfRect reyes = new MatOfRect();

        boolean isLeftFound = false;
        boolean isRightFound = false;
        String state = "no state";
        //parameter list for multidetect:

            /*
                       image - Matrix of the type CV_8U containing an image where objects are detected.
                       objects - Vector of rectangles where each rectangle contains the detected object.
                       rejectLevels - a rejectLevels
                       levelWeights - a levelWeights
                       scaleFactor - Parameter specifying how much the image size is reduced at each image scale.
                                     Add: The scaleFactor parameter is used to determine how many different sizes
                                     of something the function will look for. Usually this value is 1.1 for the best detection.
                                     Setting this parameter to 1.2 or 1.3 will detect eyes/faces faster but doesn't find them as often,
                                     meaning the accuracy goes down.
                                     Add2: Parameter specifying how much the image size is reduced at each image scale.
                                     Basically the scale factor is used to create your scale pyramid. More explanation can be found here.
                                     In short, as described here, your model has a fixed size defined during training, which is visible in the xml.
                                     This means that this size of face is detected in the image if occuring. However, by rescaling the input image,
                                     you can resize a larger face towards a smaller one, making it detectable for the algorithm.
                                         here: https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
                                     Add3: In fact, you want it as high as possible while still getting "good" results, and this must be determined
                                     somewhat empirically. It's heavily dependent on the target to be detected, the type of cascade and the training;
                                     Even and a value as high as 1.1 for a 24x24 FD cascade has worked for me in the past. A too low value for either
                                     scaleFactor or minSize will result in huge computational costs because many more pyramid layers need to be generated.
                                     A factor of 1.05 requires roughly double the # of layers (and >2x the time) than 1.1 does.
                       minNeighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.
                                      Add: minNeighbors is used for telling the detector how sure he should be when detected an something.
                                      Normally this value is set to 3 but if you want more reliability you can set this higher.
                                      Higher values means less accuracy but more reliability.
                                      Add2: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
                                      This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality.
                                      3~6 is a good value for it.
                       flags - Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
                               Add: The flags are used for setting specific preferences, like looking for the largest object or skipping regions.
                               Default this value = 0. Setting this value can make the detection go faster.
                       minSize - Minimum possible object size. Objects smaller than that are ignored.
                                 Add: This parameter determine how small size you want to detect. You decide it! Usually, [30, 30] is a good start for face detections.
                       maxSize - Maximum possible object size. Objects larger than that are ignored.
                                 Add: This parameter determine how big size you want to detect. Again, you decide it!
                                 Usually, you don't need to set it manually, which means you want to detect any big, i.e. you don't want to miss any one that is big enough.
                       outputRejectLevels - a outputRejectLevels

            */

            //implementation:

            /*          cascadeClassifier.detectMultiScale(Mat image, MatOfRect objects,MatOfInt rejectLevels,
                                                                MatOfDouble levelWeights,double scaleFactor,int minNeighbors,int flags,
                                                                        Size minSize,Size maxSize,boolean outputRejectLevels)
            */

            //helpful links:
            /*  http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html
                http://docs.opencv.org/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html

                goal:
                http://romanhosek.cz/android-eye-detection-and-tracking-with-opencv/
                https://github.com/bsdnoobz/opencv-code/blob/master/eye-tracking.cpp

                aditional: how to build cascade: https://www.cs.auckland.ac.nz/~m.rezaei/Tutorials/
            */
        // Use the classifier to detect faces

        if (cascadeClassifierFace != null) {
            //faces
            cascadeClassifierFace.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }

        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();

        //check if face detected
        if((facesArray.length)>0) {
            Core.rectangle(aInputFrame, facesArray[0].tl(), facesArray[0].br(), new Scalar(0, 255, 0, 255), 3);

            Mat ROI = grayscaleImage.submat(facesArray[0]);
            // Use the classifier to detect eyes
            //eyes
            if (cascadeClassifierEyes != null)
                cascadeClassifierEyes.detectMultiScale(ROI, eyes, 1.1, 2, 2,
                        new Size(30, 30), new Size());
            //if any eyes found
            Rect[] eyesArray = eyes.toArray();

            if((eyesArray.length)>0) {
                eyesArray[0].x = eyesArray[0].x + facesArray[0].x;
                eyesArray[0].y = eyesArray[0].y + facesArray[0].y;
                Core.rectangle(aInputFrame, eyesArray[0].tl(), eyesArray[0].br(), new Scalar(255, 0, 255, 0), 3);

                Mat eyesROI = grayscaleImage.submat(eyesArray[0]);

                Rect rectRight = new Rect();
                rectRight = eyesArray[0].clone();
                rectRight.width /= 2;
                Rect rectLeft = new Rect();
                rectLeft = rectRight.clone();
                rectLeft.x += rectRight.width;
                Mat leftROI = grayscaleImage.submat(rectLeft);
                Mat rightROI = grayscaleImage.submat(rectRight);

                if (cascadeClassifierREye != null) {
                    //faces
                    cascadeClassifierREye.detectMultiScale(rightROI, reyes, 1.1, 2, 2,
                            new Size(20, 20), new Size());
                    // If there are any right eyes found, draw a rectangle around it
                    Rect[] reyesArray = reyes.toArray();

                    if(reyesArray.length > 0 ){
                        isRightFound = true;
                        reyesArray[0].x = reyesArray[0].x + eyesArray[0].x;
                        reyesArray[0].y = reyesArray[0].y + eyesArray[0].y;
                        Core.rectangle(aInputFrame, reyesArray[0].tl(), reyesArray[0].br(), new Scalar(255, 0, 0, 0), 3);
                    }
                }

                if (cascadeClassifierLEye != null) {
                    //faces
                    cascadeClassifierLEye.detectMultiScale(leftROI, leyes, 1.1, 2, 2,
                            new Size(20, 20), new Size());

                    // If there are any left eyes found, draw a rectangle around it
                    Rect[] leyesArray = leyes.toArray();
                    if (leyesArray.length > 0) {
                        isLeftFound = true;
                        leyesArray[0].x = leyesArray[0].x + eyesArray[0].x + rectRight.width;
                        leyesArray[0].y = leyesArray[0].y + eyesArray[0].y;
                        Core.rectangle(aInputFrame, leyesArray[0].tl(), leyesArray[0].br(), new Scalar(0, 0, 255, 0), 3);
                    }
                }

                if(!isLeftFound && !isRightFound){
                    bothCounter ++;
                    if(bothCounter >= DETECT_THRESHHOLD) {
                        state = "both";
                    }
                }else if(!isLeftFound){
                    leftCounter ++;
                    if(leftCounter >= DETECT_THRESHHOLD) {
                        state = "left";
                    }
                }else if(!isRightFound){
                    rightCounter ++;
                    if(rightCounter >= DETECT_THRESHHOLD) {
                        state = "right";
                    }
                }else{
                    rightCounter = 0;
                    leftCounter = 0;
                    bothCounter = 0;
                }
            }
        }

        Core.putText(aInputFrame,state,new Point(100,100),3,1,new Scalar(255,0,255,0),2);
        return aInputFrame;
    }
    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }
}