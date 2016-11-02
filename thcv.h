#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread>
#include <vector>

class MyCap {
public:
    void run();
    static void RunThreads(MyCap * cap);
    bool keepRunning = true; // Will be changed by the external program.
    std::vector<std::thread> capThreads;
private:
    cv::Mat frame;
    cv::VideoCapture cap;
    MyCap() { }
    static MyCap * s_instance;
public:
    static MyCap *instance();
};