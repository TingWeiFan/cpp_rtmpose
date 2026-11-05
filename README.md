# 簡化版 RTMPose C++

檔案分工：

- `include/rtmpose.h`：公開功能，像目錄或操作面板。
- `src/rtmpose.cpp`：TensorRT、CUDA、前後處理的實作細節。
- `src/main.cpp`：應用程式流程，只負責讀圖、呼叫模型、存圖。
- `CMakeLists.txt`：告訴 CMake 如何組合及連結以上檔案。

建置：

```bash
cd /home/ubuntu20/workspace/mmdeploy/cpp_rtmpose/simple
cmake -S . -B build
cmake --build build -j
```

執行：

```bash
./build/rtmpose_app \
  /home/ubuntu20/workspace/mmdeploy/rtmpose-trt/rtmpose-m/end2end-fp16.engine \
  /home/ubuntu20/workspace/mmdeploy/demo/resources/human-pose.jpg \
  rtmpose_cpp_result.jpg
```

