#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// 一個人體關鍵點的結果。
struct Keypoint {
    std::string name;
    cv::Point2f point;
    float score;
};

// main.cpp 只需要知道這個類別提供哪些功能。
// TensorRT、CUDA 和 SimCC 的複雜細節都放在 rtmpose.cpp。
class RTMPose {
public:
    explicit RTMPose(const std::string& engine_path);
    ~RTMPose();

    RTMPose(const RTMPose&) = delete;
    RTMPose& operator=(const RTMPose&) = delete;

    // 輸入一張 OpenCV BGR 圖片，回傳 17 個人體關鍵點。
    std::vector<Keypoint> infer(const cv::Mat& image);

    // 把骨架和關鍵點畫在圖片上。
    void draw(cv::Mat& image, const std::vector<Keypoint>& keypoints,
              float score_threshold = 0.3F) const;

private:
    // Impl 的內容只存在於 rtmpose.cpp，讓這個 header 保持乾淨。
    class Impl;
    std::unique_ptr<Impl> impl;
};
