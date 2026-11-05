#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "inputUtils.h"
#include "rtmpose.h"

namespace {

    bool isCameraIndex(const std::string& input) {
        return !input.empty() &&
               std::all_of(input.begin(), input.end(),
                           [](unsigned char character) { return std::isdigit(character); });
    }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "使用方式：\n"
                  << "./rtmpose_app MODEL.engine INPUT [OUTPUT]\n";
        return 1;
    }

    const std::string engine_path = argv[1];
    const std::string input_path = argv[2];
    const InputType input_type = detectInputType(input_path);

    if (input_type == InputType::UNKNOWN) {
        std::cerr << "無法判斷輸入類型：" << input_path << '\n';
        return 1;
    }

    try {
        // 模型只載入一次，影片的每一幀重複使用同一個模型。
        RTMPose model(engine_path);

        if (input_type == InputType::IMAGE) {
            const std::string output_path = argc >= 4 ? argv[3] : "rtmpose_cpp_result.jpg";

            cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "無法讀取圖片：" << input_path << '\n';
                return 1;
            }

            const std::vector<Keypoint> keypoints = model.infer(image);
            model.draw(image, keypoints);

            if (!cv::imwrite(output_path, image)) {
                std::cerr << "無法儲存圖片：" << output_path << '\n';
                return 1;
            }

            std::cout << "結果圖片：" << output_path << '\n';
            return 0;
        }

        cv::VideoCapture capture;
        if (isCameraIndex(input_path)) {
            capture.open(std::stoi(input_path)); // 開啟本機攝影機
        } else {
            capture.open(input_path); // 開啟影片檔案或網路串流
        }

        if (!capture.isOpened()) {
            std::cerr << "無法打開影片或串流：" << input_path << '\n';
            return 1;
        }

        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            std::cerr << "無法取得第一幀畫面\n";
            return 1;
        }

        double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0) {
            fps = 30.0;
        }

        const std::string output_path = argc >= 4 ? argv[3] : "rtmpose_result.mp4";
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
        if (!writer.isOpened()) {
            std::cerr << "無法建立輸出影片：" << output_path << '\n';
            return 1;
        }

        do {
            const std::vector<Keypoint> keypoints = model.infer(frame);
            model.draw(frame, keypoints);
            writer.write(frame);
        } while (capture.read(frame));

        std::cout << "結果影片：" << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "錯誤：" << error.what() << '\n';
        return 1;
    }
}
