#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

#include <opencv2/opencv.hpp>
#include <iostream>

float invoke_kernel(unsigned char *img, int w, int h, int window_size, int size);

int main(int argc, char *argv[])
{
    int kernel_size = 3;
    if (argc > 1)
    {
        kernel_size = std::stoi(argv[1]);
    }

    cv::VideoCapture cap("caterpillar.mp4");

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    // Get the frame rate of the video
    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = 1000 / fps; // Calculate the delay between frames in milliseconds

    // Create a window to display the frames
    cv::namedWindow("Blurred Video", cv::WINDOW_NORMAL);

    while (true)
    {
        cv::Mat frame;
        bool ret = cap.read(frame); // Read the next frame

        if (!ret)
        {
            std::cout << "End of video" << std::endl;
            break;
        }

        cv::Mat flatMat = frame.reshape(1, frame.total() * frame.channels());
        uchar *data = flatMat.ptr<uchar>(0);

        float ms = invoke_kernel(data, frame.rows, frame.cols, kernel_size, frame.total() * frame.channels());

        cv::Mat hwc_img = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);

        for (int i = 0; i < frame.rows; i++)
        {
            for (int j = 0; j < frame.cols; j++)
            {
                cv::Vec3b color = cv::Vec3b(0, 0, 0);
                for (int k = 0; k < frame.channels(); k++)
                {
                    color[k] = data[(i * frame.cols + j) * frame.channels() + k];
                }

                hwc_img.at<cv::Vec3b>(i, j) = color;
            }
        }

        cv::imshow("Blurred Video", hwc_img);

        printf("Frame took %f ms\n", ms);
        if (cv::waitKey(delay) == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}