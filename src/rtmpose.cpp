#include "rtmpose.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

namespace {

    constexpr int kInputWidth = 288;
    constexpr int kInputHeight = 384;
    constexpr float kPadding = 1.25F;
    constexpr float kSplitRatio = 2.0F;

    const std::array<const char*, 17> kKeypointNames = {
        "nose",           "left_eye",   "right_eye",   "left_ear",   "right_ear",   "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
        "right_hip",      "left_knee",  "right_knee",  "left_ankle", "right_ankle"};

    const int kSkeleton[][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                                {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
                                {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TensorRT] " << message << '\n';
            }
        }
    };

    void checkCuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + "：" + cudaGetErrorString(status));
        }
    }

    std::vector<char> readBinaryFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);  // 打開檔案
        if (!file) {
            throw std::runtime_error("無法開啟 engine：" + path);
        }

        const std::streamsize size = file.tellg();
        if (size <= 0) {
            throw std::runtime_error("engine 檔案是空的：" + path);
        }

        file.seekg(0, std::ios::beg);
        std::vector<char> data(static_cast<std::size_t>(size));
        // 把資料放到 data 裡
        if (!file.read(data.data(), size)) {
            throw std::runtime_error("讀取 engine 失敗：" + path);
        }
        return data;
    }

    // 計算元素數量
    std::size_t volume(const nvinfer1::Dims& dims) {
        std::size_t result = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0) {
                throw std::runtime_error("Tensor 維度尚未確定");
            }
            result *= static_cast<std::size_t>(dims.d[i]);
        }
        return result;
    }

    cv::Point2f thirdPoint(const cv::Point2f& first, const cv::Point2f& second) {
        const cv::Point2f direction = first - second;
        return second + cv::Point2f(-direction.y, direction.x);
    }

    cv::Mat makeAffine(const cv::Point2f& center, const cv::Point2f& scale, bool inverse) {
        cv::Point2f source[3];
        cv::Point2f destination[3];

        source[0] = center;
        source[1] = center + cv::Point2f(0.0F, -0.5F * scale.x);
        source[2] = thirdPoint(source[0], source[1]);

        destination[0] = cv::Point2f(kInputWidth * 0.5F, kInputHeight * 0.5F);
        destination[1] = cv::Point2f(kInputWidth * 0.5F, kInputHeight * 0.5F - kInputWidth * 0.5F);
        destination[2] = thirdPoint(destination[0], destination[1]);

        return inverse ? cv::getAffineTransform(destination, source)
                       : cv::getAffineTransform(source, destination);
    }

    struct PreparedInput {
        std::vector<float> values;
        cv::Mat inverse_affine;
    };

    PreparedInput preprocess(const cv::Mat& image) {
        const cv::Point2f center(image.cols * 0.5F, image.rows * 0.5F);
        cv::Point2f scale(image.cols * kPadding, image.rows * kPadding);

        const float target_aspect = static_cast<float>(kInputWidth) / kInputHeight;
        if (scale.x > scale.y * target_aspect) {
            scale.y = scale.x / target_aspect;
        } else {
            scale.x = scale.y * target_aspect;
        }

        const cv::Mat affine = makeAffine(center, scale, false);
        const cv::Mat inverse_affine = makeAffine(center, scale, true);

        cv::Mat crop;
        cv::warpAffine(image, crop, affine, cv::Size(kInputWidth, kInputHeight), cv::INTER_LINEAR);
        cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
        crop.convertTo(crop, CV_32FC3);

        constexpr std::array<float, 3> mean = {123.675F, 116.28F, 103.53F};
        constexpr std::array<float, 3> stddev = {58.395F, 57.12F, 57.375F};

        // 建立模型輸入所需要的記憶體
        const std::size_t plane = static_cast<std::size_t>(kInputWidth * kInputHeight);
        std::vector<float> values(3 * plane);

        for (int y = 0; y < kInputHeight; ++y) {
            const cv::Vec3f* row = crop.ptr<cv::Vec3f>(y);
            for (int x = 0; x < kInputWidth; ++x) {
                const std::size_t position = static_cast<std::size_t>(y * kInputWidth + x);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    values[channel * plane + position] =
                        (row[x][channel] - mean[channel]) / stddev[channel];
                }
            }
        }

        return {std::move(values), inverse_affine};  // move 會把記憶體交給新的 vector
    }

}  // namespace

class RTMPose::Impl {
public:
    ~Impl() {
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (device_simcc_y != nullptr) cudaFree(device_simcc_y);
        if (device_simcc_x != nullptr) cudaFree(device_simcc_x);
        if (device_input != nullptr) cudaFree(device_input);
        delete context;
        delete engine;
        delete runtime;
    }

    void load(const std::string& engine_path) {
        const std::vector<char> engine_data = readBinaryFile(engine_path);

        runtime = nvinfer1::createInferRuntime(logger);
        if (runtime == nullptr) {
            throw std::runtime_error("建立 TensorRT Runtime 失敗");
        }

        engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (engine == nullptr) {
            throw std::runtime_error("載入 TensorRT engine 失敗");
        }

        context = engine->createExecutionContext();
        if (context == nullptr) {
            throw std::runtime_error("建立 ExecutionContext 失敗");
        }

        input_index = engine->getBindingIndex("input");
        simcc_x_index = engine->getBindingIndex("simcc_x");
        simcc_y_index = engine->getBindingIndex("simcc_y");
        if (input_index < 0 || simcc_x_index < 0 || simcc_y_index < 0) {
            throw std::runtime_error("找不到 input/simcc_x/simcc_y binding");
        }

        if (engine->getBindingDataType(input_index) != nvinfer1::DataType::kFLOAT ||
            engine->getBindingDataType(simcc_x_index) != nvinfer1::DataType::kFLOAT ||
            engine->getBindingDataType(simcc_y_index) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("這個簡化版本只支援 FP32 輸入輸出");
        }

        const nvinfer1::Dims4 input_dims(1, 3, kInputHeight, kInputWidth);
        if (!context->setBindingDimensions(input_index, input_dims) ||
            !context->allInputDimensionsSpecified()) {
            throw std::runtime_error("設定輸入形狀失敗");
        }

        simcc_x_dims = context->getBindingDimensions(simcc_x_index);
        simcc_y_dims = context->getBindingDimensions(simcc_y_index);
        input_count = static_cast<std::size_t>(3 * kInputHeight * kInputWidth);
        simcc_x_count = volume(simcc_x_dims);
        simcc_y_count = volume(simcc_y_dims);

        checkCuda(cudaMalloc(&device_input, input_count * sizeof(float)),
                  "配置輸入 GPU 記憶體失敗");
        checkCuda(cudaMalloc(&device_simcc_x, simcc_x_count * sizeof(float)),
                  "配置 simcc_x GPU 記憶體失敗");
        checkCuda(cudaMalloc(&device_simcc_y, simcc_y_count * sizeof(float)),
                  "配置 simcc_y GPU 記憶體失敗");
        checkCuda(cudaStreamCreate(&stream), "建立 CUDA stream 失敗");
    }

    std::vector<Keypoint> infer(const cv::Mat& image) {
        if (image.empty()) {
            throw std::invalid_argument("輸入圖片是空的");
        }

        const PreparedInput prepared = preprocess(image);
        std::vector<float> simcc_x(simcc_x_count);
        std::vector<float> simcc_y(simcc_y_count);

        // cudaMemcpyAsync: 在 CPU 記憶體和 GPU 記憶體之間複製資料
        checkCuda(cudaMemcpyAsync(device_input, prepared.values.data(), input_count * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "圖片複製到 GPU 失敗");

        std::vector<void*> bindings(static_cast<std::size_t>(engine->getNbBindings()), nullptr);
        bindings[static_cast<std::size_t>(input_index)] =
            device_input;  // 告訴 TensorRT 這個地址的用途
        bindings[static_cast<std::size_t>(simcc_x_index)] = device_simcc_x;
        bindings[static_cast<std::size_t>(simcc_y_index)] = device_simcc_y;

        if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
            throw std::runtime_error("TensorRT 推論失敗");
        }

        checkCuda(cudaMemcpyAsync(simcc_x.data(), device_simcc_x, simcc_x_count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream),
                  "simcc_x 複製回 CPU 失敗");
        checkCuda(cudaMemcpyAsync(simcc_y.data(), device_simcc_y, simcc_y_count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream),
                  "simcc_y 複製回 CPU 失敗");
        checkCuda(cudaStreamSynchronize(stream), "等待 GPU 完成失敗");

        const int keypoint_count = simcc_x_dims.d[1];
        const int x_bins = simcc_x_dims.d[2];
        const int y_bins = simcc_y_dims.d[2];
        std::vector<cv::Point2f> crop_points;
        std::vector<float> scores;
        crop_points.reserve(static_cast<std::size_t>(keypoint_count));  // reserve: 預先保留空間
        scores.reserve(static_cast<std::size_t>(keypoint_count));

        for (int keypoint = 0; keypoint < keypoint_count; ++keypoint) {
            const float* x_begin = simcc_x.data() + keypoint * x_bins;
            const float* y_begin = simcc_y.data() + keypoint * y_bins;
            const float* x_max = std::max_element(x_begin, x_begin + x_bins);
            const float* y_max = std::max_element(y_begin, y_begin + y_bins);

            // 建立座標點
            crop_points.emplace_back(static_cast<float>(x_max - x_begin) / kSplitRatio,
                                     static_cast<float>(y_max - y_begin) / kSplitRatio);
            scores.push_back(std::min(*x_max, *y_max));
        }

        std::vector<cv::Point2f> image_points;
        cv::transform(crop_points, image_points, prepared.inverse_affine);

        std::vector<Keypoint> result;
        result.reserve(image_points.size());
        for (std::size_t i = 0; i < image_points.size(); ++i) {
            result.push_back({kKeypointNames.at(i), image_points[i], scores[i]});
        }
        return result;
    }

private:
    // 接收 TensorRT 的錯誤和警告訊息。
    Logger logger;

    // Runtime 負責把硬碟中的 .engine 資料還原成 TensorRT 模型。
    // 一開始設為 nullptr，代表尚未建立。
    nvinfer1::IRuntime* runtime{nullptr};

    // Engine 是載入完成的模型，保存網路結構、權重和最佳化資訊。
    nvinfer1::ICudaEngine* engine{nullptr};

    // Context 管理「這一次推論」的狀態，真正執行 enqueueV2()。
    nvinfer1::IExecutionContext* context{nullptr};

    // 三個 binding 在 TensorRT bindings 陣列中的位置。
    // -1 代表還沒找到；load() 會用名稱查出真正的編號。
    int input_index{-1};
    int simcc_x_index{-1};
    int simcc_y_index{-1};

    // 兩個輸出的形狀，例如 simcc_x 是 [1, 17, 576]。
    // 需要形狀才能知道輸出總共有多少個數值。
    nvinfer1::Dims simcc_x_dims{};
    nvinfer1::Dims simcc_y_dims{};

    // 輸入和輸出的 float 元素數量，不是 byte 數量。
    // 配置記憶體時還要再乘上 sizeof(float)。
    std::size_t input_count{0};
    std::size_t simcc_x_count{0};
    std::size_t simcc_y_count{0};

    // GPU 記憶體地址：一塊放輸入，另外兩塊接收模型輸出。
    // load() 使用 cudaMalloc() 配置，解構函式使用 cudaFree() 釋放。
    void* device_input{nullptr};
    void* device_simcc_x{nullptr};
    void* device_simcc_y{nullptr};

    // GPU 工作隊伍。輸入複製、推論、輸出複製會依序排進同一條 stream。
    cudaStream_t stream{nullptr};
};

RTMPose::RTMPose(const std::string& engine_path) : impl(new Impl()) { impl->load(engine_path); }

RTMPose::~RTMPose() = default;

std::vector<Keypoint> RTMPose::infer(const cv::Mat& image) { return impl->infer(image); }

void RTMPose::draw(cv::Mat& image, const std::vector<Keypoint>& keypoints,
                   float score_threshold) const {
    for (const auto& edge : kSkeleton) {
        const Keypoint& first = keypoints.at(static_cast<std::size_t>(edge[0]));
        const Keypoint& second = keypoints.at(static_cast<std::size_t>(edge[1]));
        if (first.score >= score_threshold && second.score >= score_threshold) {
            cv::line(image, first.point, second.point, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
    }

    for (const Keypoint& keypoint : keypoints) {
        if (keypoint.score >= score_threshold) {
            cv::circle(image, keypoint.point, 4, cv::Scalar(0, 80, 255), -1, cv::LINE_AA);
        }
    }
}
