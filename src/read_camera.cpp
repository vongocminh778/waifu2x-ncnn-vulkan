#include <stdio.h>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

// Include stb_image and stb_image_write headers
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace cv;

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"
#include "waifu2x.h"

// Helper function to get file extension
std::string get_file_extension(const std::string& filepath) {
    size_t dot_pos = filepath.find_last_of(".");
    if (dot_pos == std::string::npos) return "";
    return filepath.substr(dot_pos + 1);
}

// Function to print image info
void print_image_info(const ncnn::Mat& image, const std::string& name) {
    std::cout << "Image: " << name << std::endl;
    std::cout << "Width: " << image.w << ", Height: " << image.h << ", Channels: " << image.elempack << std::endl;

    std::cout << "First few pixel values: ";
    for (int i = 0; i < std::min(15, image.w * image.h * image.elempack); ++i) {
        std::cout << static_cast<int>(reinterpret_cast<const unsigned char*>(image.data)[i]) << " ";
    }
    std::cout << std::endl;
}

// Function to print OpenCV image info
void print_cv_image_info(const cv::Mat& image, const std::string& name) {
    std::cout << "Image: " << name << std::endl;
    std::cout << "Width: " << image.cols << ", Height: " << image.rows << ", Channels: " << image.channels() << std::endl;

    std::cout << "First few pixel values: ";
    for (int i = 0; i < std::min(15, image.cols * image.rows * image.channels()); ++i) {
        std::cout << static_cast<int>(image.data[i]) << " ";
    }
    std::cout << std::endl;
}

// Function to calculate tilesize based on GPU heap budget
int calculate_tilesize(int gpuid) {
    uint32_t heap_budget = ncnn::get_gpu_device(gpuid)->get_heap_budget();
    if (heap_budget > 2600) return 400;
    if (heap_budget > 740) return 200;
    if (heap_budget > 250) return 100;
    return 32; // Default
}

int main(int argc, char** argv) {
    // const char* parampath = "../../models-upconv_7_anime_style_art_rgb/noise3_scale2.0x_model.param";
    // const char* modelpath = "../../models-upconv_7_anime_style_art_rgb/noise3_scale2.0x_model.bin";

    const char* parampath = "../../models-upconv_7_photo/noise3_scale2.0x_model.param";
    const char* modelpath = "../../models-upconv_7_photo/noise3_scale2.0x_model.bin";
    int prepadding = 7; // Default padding for SRMD

    // const char* parampath = "../../models-cunet/noise3_scale2.0x_model.param";
    // const char* modelpath = "../../models-cunet/noise3_scale2.0x_model.bin";
    // int prepadding = 0; // Default padding for SRMD
    
    // Initialize SRMD
    ncnn::create_gpu_instance();
    int gpuid = ncnn::get_default_gpu_index();
    Waifu2x waifu2x(gpuid, 0);

    int noise = 3;
    int scale = 2;
    
    int tilesize = calculate_tilesize(gpuid);

    if (waifu2x.load(parampath, modelpath) != 0) {
        std::cerr << "Failed to load model: " << parampath << " and " << modelpath << std::endl;
        return -1;
    }

    waifu2x.noise = noise;
    waifu2x.scale = scale;
    waifu2x.prepadding = prepadding;
    waifu2x.tilesize = tilesize;

    // Open webcam
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open webcam" << std::endl;
        return -1;
    }

    Mat frame, beforeImage;
    double fps = 0.0;
    double tick_frequency = getTickFrequency();
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }

        // Convert frame from BGR to RGB
        Mat frame_rgb;
        cvtColor(frame, frame_rgb, COLOR_BGR2RGB);

        // // Crop 256x256 region around the center
        // int cropSize = 256;
        // int centerX = frame_rgb.cols / 2;
        // int centerY = frame_rgb.rows / 2;
        // Rect cropRegion(centerX - cropSize / 2, centerY - cropSize / 2, cropSize, cropSize);
        // Mat croppedImage = frame_rgb(cropRegion);

        resize(frame_rgb, beforeImage, Size(frame_rgb.cols * scale, frame_rgb.rows * scale), INTER_LINEAR);
        
        Mat img_process = frame_rgb.clone();
        ncnn::Mat cv_inimage(frame_rgb.cols, frame_rgb.rows, img_process.data, (size_t)frame_rgb.channels(), frame_rgb.channels());

        ncnn::Mat ncnn_outimage(frame_rgb.cols * scale, frame_rgb.rows * scale, (size_t)frame_rgb.channels(), frame_rgb.channels());

        int64 frame_tick = getTickCount();

        // Process the image
        waifu2x.process(cv_inimage, ncnn_outimage);

        int64 end_tick = getTickCount();
        double frame_time = (end_tick - frame_tick) / tick_frequency;
        fps = 1.0 / frame_time;

        // Convert ncnn::Mat back to cv::Mat
        Mat output(ncnn_outimage.h, ncnn_outimage.w, CV_8UC3, (unsigned char*)ncnn_outimage.data);
        cvtColor(output, output, COLOR_RGB2BGR);
        cvtColor(beforeImage, beforeImage, COLOR_RGB2BGR);

        // Display FPS on frame
        putText(beforeImage, "FPS: " + std::to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        // Display the processed frame
        imshow("Before Frame", beforeImage);
        imshow("After Frame", output);

        // Break the loop on 'q' key press
        if (waitKey(1) == 'q') break;
    }

    return 0;
}