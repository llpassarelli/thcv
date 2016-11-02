/*

developer: llpassarelli@gmail.com - www.passarelliautomacao.com.br - Brasil - 2016

*/

#include <iostream>
#include <string.h>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

std::mutex mtx;

pthread_mutex_t frameLocker;
Mat frame  = Mat::zeros(Size(640,480),CV_8UC3);    
Mat result = Mat::zeros(Size(640,480),CV_8UC3);
string resultado = "";  

VideoCapture capture;


void inspect(Mat &mask, Mat &frame)
{    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    try
    {   
        ///Find the contours
        findContours( mask.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0) );
        if (!contours.empty() && !hierarchy.empty())
        {
            // iterate through all the top-level contours,
            int idx = 0, largestComp = 0;
            double maxArea = 0;
            vector<Point> current_contour;
            for( ; idx >= 0; idx = hierarchy[idx][0] )
            {
                const vector<Point>& c = contours[idx];
                double area = fabs(contourArea(Mat(c)));

                if( area > maxArea )
                {
                    maxArea = area;
                    largestComp = idx;
                }
            }

            current_contour=contours[largestComp];
            int cnt_len = arcLength(current_contour, 1);
            //00.0036
            approxPolyDP( Mat(contours[largestComp]), current_contour, 0.0036 * cnt_len, true );
            int cnt_area = contourArea(current_contour);
            int cnt_convex = isContourConvex(current_contour);
            int cnt_size = current_contour.size();
            
            resultado = "sz: " + to_string(cnt_size) + " cvx: " + to_string(cnt_convex) + " area: " + to_string(cnt_area) + " per: " + to_string(cnt_len);

            Scalar color = Scalar( 0, 0, 255 );
            bool pass = false;
            
            
            if ( cnt_size > 6 and cnt_area > 60000 and cnt_area < 80000 and cnt_convex ){
                color = Scalar( 0, 250, 0 );
                pass = true;

                //string center = to_string( minRect[0].center() );
                //string size = to_string( minRect[0].size() );
               
            }
           
            // rotated rectangle
            vector<RotatedRect> minRect( contours.size() );
            minRect[0] = minAreaRect( Mat(current_contour) );
            Point2f rect_points[4]; minRect[0].points( rect_points );
            for( int j = 0; j < 4; j++ )
                line( result, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
            

            //mean
            Scalar tempMean = mean(frame, mask);
            int r = tempMean.val[0];
            int g = tempMean.val[1];
            int b = tempMean.val[2];
       
            resultado += " mean: [" + to_string(r) + ", " + to_string(g) + ", " + to_string(b) + "] w: " + to_string(int(minRect[0].size.width)) + " h: " + to_string(int(minRect[0].size.height));
            //cout << resultado << "\n";
            drawContours( result, contours, largestComp, color, 1, LINE_8, hierarchy );
            cout << resultado << "\n";
        }
    }
    catch (Exception &e)
    {   
        cout << e.msg << endl;
        return;
    }
}

void processing(Mat &frame)
{
    try
    {
        vector<Mat> bgr_planes;
        split( frame, bgr_planes );
        //Mat b = bgr_planes[0];
        //Mat g = bgr_planes[1];
        Mat r = bgr_planes[2];
        Mat mask   = Mat::zeros( frame.size(), CV_8UC1 );
        Mat res    = Mat::zeros( frame.size(), CV_8UC1 );
        Mat open   = Mat::zeros( frame.size(), CV_8UC1 );
        Mat close  = Mat::zeros( frame.size(), CV_8UC1 );
        Mat thresh = Mat::zeros( frame.size(), CV_8UC1 );
        resize( r, mask, Size(), 0.5, 0.5, INTER_LINEAR );
        threshold(mask,thresh, 120, 255, THRESH_BINARY);
        Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(thresh,open, MORPH_OPEN, kernel1);
        morphologyEx(open,close, MORPH_CLOSE, kernel1);
        GaussianBlur(close,res, Size(3, 3), 3);
        resize( res, res, Size(), 2, 2, INTER_LINEAR );
        inspect(res, frame);
    }
    catch (Exception &e)
    {   
         cout << e.msg << endl;
    }

}

float trigger(Mat &tframe)
{
    try
    {
        float resmean = 0;
        int x=200,y=200,w=300,h=110;
        Rect roi = Rect(x,y,w,h);
        Mat mask0 = Mat::zeros( tframe.size(), CV_8UC1 );
        rectangle(mask0,roi,255,1);
        //rectangle(frame,roi,255,1);
        Mat imRoi = frame(roi);
        //Mat mask1 = (imRoi != 0);
        Scalar tempMean = mean(imRoi);
        resmean = tempMean.val[0];
        if (resmean < 123)
            cout << "\n trigger mean: " << resmean << "\n";
        return resmean;
    
    }
    catch (Exception &e)
    {   
         cout << e.msg << endl;
         return 0;
    }
}

//capture
void *UpdateFrame(void *arg)
{
    for(;;)
    {
        try 
        {
            Mat tempFrame = Mat::zeros( frame.size(), CV_8UC3 );
            capture >> tempFrame;
            
            pthread_mutex_lock(&frameLocker);
            if (!tempFrame.empty())
                frame = tempFrame;
            pthread_mutex_unlock(&frameLocker);
        }
        catch(Exception &e)
        {
            cout << e.msg << endl;
        }
    }   
}



int main(int, char**)
{
    // video 1, exposure manual, gain manual 
    int status = system( "v4l2-ctl -d /dev/video1 -c gain_automatic=0 -c gain=0 -c auto_exposure=1 -c exposure=30 -c power_line_frequency=1" );
    //int status = system("v4l2-ctl -d /dev/video1 -C gain_automatic -C gain -C auto_exposure -C exposure");
    double start=0;
    double end=0;
    capture.open(1);
    pthread_mutex_init(&frameLocker, NULL);
    pthread_t UpdateThread;
    pthread_create(&UpdateThread, NULL, UpdateFrame, NULL);
    
    for (;;) 
    {
        try
        {
            Mat currentFrame = Mat::zeros(Size(640,480),CV_8UC3); 
            //get current frame
            pthread_mutex_lock(&frameLocker);
            currentFrame=frame;
            pthread_mutex_unlock(&frameLocker);

            if (currentFrame.empty()){
                cout << "frame empty" << "\n";
                continue;
            }

            start = double(getTickCount());
            // trigger
            float mean = 0;
            pthread_mutex_lock(&frameLocker);
            mean = trigger(currentFrame);

            if (mean > 123)
            {  
                if (!currentFrame.empty()){ 
                    result = currentFrame.clone();
                    processing(currentFrame);
                    
                    if (!result.empty()){
                        currentFrame = result;
                        int duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
                        string text = to_string(duration_ms) + " ms";
                        putText(currentFrame, text, Point(10, 20), 1, 1,Scalar(255, 255, 255),1, LINE_AA); // line thickness and type
                        putText(currentFrame, resultado, Point(10, 460), 1, 1,Scalar(255, 255, 255),1, LINE_AA);

                    }
                }
            }
            pthread_mutex_unlock(&frameLocker);
            //mean decor rect
            int x=200,y=200,w=300,h=110;
            Rect roi = Rect(x,y,w,h);
            rectangle(currentFrame,roi,255,1);
            imshow("Passarelli Automacao", currentFrame);
            waitKey(1);
            
        }
        catch(Exception &e)
        {
            cout << e.msg << endl;
        }

    }
    
    system("pause");  
    return 0;
}