# RTMPose TensorRT C++ 推論

以 C++ 直接呼叫 TensorRT，載入 RTMPose 的 FP16 engine，完成單張影像的人體姿態估算，並把關鍵點畫在輸出影像上。

這個專案的目的是把 Python 端的 MMPose / MMDeploy 推論流程改寫成 C++，自己實作前處理、推論和後處理，不依賴 MMDeploy SDK，了解部署時每一步實際在做什麼。

<!-- 待補：放一張輸出結果圖，例如 ![result](docs/rtmpose_cpp_result.jpg) -->

## 推論流程

```
讀取影像 → 前處理 → TensorRT 推論 → 後處理 → 繪製關鍵點並存檔
```

| 步驟 | 內容 |
|---|---|
| 前處理 | 仿射變換至模型輸入尺寸（〈待補，例如 192×256〉）、BGR 轉 RGB、mean/std 正規化、HWC 轉 CHW |
| 推論 | 載入 `.engine`，配置 GPU 記憶體，以 CUDA stream 執行推論 |
| 後處理 | 解碼 SimCC 輸出（x、y 各取 argmax），以反仿射變換轉回原圖座標 |

## 專案結構

```
.
├── CMakeLists.txt      # 建置設定，連結 TensorRT、CUDA、OpenCV
├── include/
│   └── rtmpose.h       # 對外介面：載入模型、執行推論
└── src/
    ├── rtmpose.cpp     # 實作：TensorRT 載入與推論、CUDA 記憶體管理、前後處理
    └── main.cpp        # 範例程式：讀圖、呼叫模型、存出結果
```

`main.cpp` 只負責應用流程，所有和 TensorRT、CUDA 有關的細節都封裝在 `rtmpose.cpp`，換成影片或其他輸入時不需要修改推論的部分。

## 環境

| 項目 | 版本 |
|---|---|
| OS | Ubuntu 20.04 |
| GPU | NVIDIA RTX 3080 |
| CUDA | 〈待補〉 |
| TensorRT | 〈待補〉 |
| OpenCV | 〈待補〉 |
| CMake | 〈待補〉 |

## 準備 engine

engine 由 MMDeploy 將 RTMPose-m 轉換為 TensorRT FP16 產生（`end2end-fp16.engine`）。

TensorRT engine 綁定建置時的 GPU 型號與 TensorRT 版本，換到不同的 GPU 或裝置（例如 Jetson）上執行時，需要在該裝置上重新轉換。

## 建置

```bash
cmake -S . -B build
cmake --build build -j
```

## 執行

```bash
./build/rtmpose_app <engine 路徑> <輸入影像> <輸出影像>
```

範例：

```bash
./build/rtmpose_app \
  models/rtmpose-m/end2end-fp16.engine \
  assets/human-pose.jpg \
  rtmpose_cpp_result.jpg
```

## 結果驗證

以同一張影像比對 C++ 與 Python（MMDeploy）的關鍵點輸出，確認前後處理實作正確。

| 指標 | 結果 |
|---|---|
| 關鍵點座標平均誤差 | 〈待補〉 px |
| 關鍵點座標最大誤差 | 〈待補〉 px |

## 推論時間

| 階段 | 時間 (ms) |
|---|---|
| 前處理 | 〈待補〉 |
| 推論 | 〈待補〉 |
| 後處理 | 〈待補〉 |
| 端到端 | 〈待補〉 |

## 待改進

- 前處理目前在 CPU 執行，可改為 CUDA kernel 在 GPU 上處理
- 目前只處理單張影像，可擴充為影片串流
- 加入人物偵測模型，支援多人姿態估算
