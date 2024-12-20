---

# VAD-ASR-SC for Android

## Overview

This project demonstrates the integration of Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Speaker Confirmation on Android devices, utilizing optimized models for high-speed performance.

## Getting Started

1. **Download the Models:**
   - The demo models are available for download [here](https://drive.google.com/drive/folders/1ErEdY6QMyJCW0yuQR03If905IhdyAHFw?usp=drive_link).

2. **Setup:**
   - After downloading, place the models into the `assets` folder.
   - Decompress the `*.so` zip file stored in the `libs/arm64-v8a` folder.

3. **Model Information:**
   - The demo models, named **FSMN-VAD**, **Paraformer**, and **ERes2Net**, are converted from ModelScope and have undergone code optimizations for extreme execution speed.
   - The inputs and outputs of these demo models differ slightly from the original versions.

4. **ONNX Runtime Adaptation:**
   - To better adapt to ONNX Runtime on Android, dynamic axes were not used during export. As a result, the exported ONNX model may not be optimal for x86_64 systems.
   - We plan to make the export method public in the future.

5. **Limitations:**
   - Due to model limitations, the English version does not perform well currently.

6. **Permissions:**
   - Ensure to approve the recorder permissions first, then relaunch the app.

7. **Performance:**
   - ASR inference takes about 25ms, speaker confirmation also takes 25ms, and VAD takes about 2ms.
   - The monitoring frequency is set to a default of 16 FPS (60ms per round) to approximate the performance of an online streaming ASR model while maintaining offline model accuracy.

8. **Features:**
   - The system can be set to recognize a custom wake word, using fuzzy matching to account for natural speech variations (Chinese wake words only). After activation, it remains active for 30 seconds by default (adjustable).
   - Simple commands can be issued directly without invoking the wake word (wake-free mode), such as "Open xxx," "Close xxx," "Navigate to xxx," "Play someone's song," etc.
   - The system can handle multi-intent judgment if sentences contain conjunctions like "and" or other continuation words.
   - Keywords like "adding voice" or "adding permission" can be spoken directly to set permissions. The same applies to "deleting permission."
   - Once permission is added, only the voice owner can modify it. The system recognizes only authorized sounds as effective commands.
   - No guarantee for the success rate of permission recognition. For more information, refer to the ERes2Net model introduction.

9. **Quantization:**
   - The quantization method for the model can be found in the folder "Do_Quantize."
   - The q4(uint4) quantization method is not recommended currently due to poor performance of the "MatMulNBits" operator in ONNX Runtime.

## Additional Resources

- Explore more projects: [https://dakeqq.github.io/overview/](https://dakeqq.github.io/overview/)

## 演示结果 Demo Results

1. 此GIF以每秒7帧的速度生成。因此，ASR看起来可能不够流畅。This GIF was generated at 7fps. Therefore, it may not look smooth enough.

![Demo Animation](https://github.com/DakeQQ/VAD-ASR-SC-for-Android/blob/main/asr.gif?raw=true)

---

# 语音活动检测 + 自动语音辨识 + 说话人确认 - 安卓

## 概述

该项目在Android设备上展示了语音活动检测（VAD）、自动语音识别（ASR）和说话人确认的集成，使用经过优化的模型以实现高速度性能。

## 快速开始

1. **下载模型：**
   - 演示模型已上传至云端硬盘：[点击这里下载](https://drive.google.com/drive/folders/1ErEdY6QMyJCW0yuQR03If905IhdyAHFw?usp=drive_link)
   - 百度链接: [点击这里](https://pan.baidu.com/s/1Si-4ebtqm2HA9omxqHCMuQ?pwd=dake) 提取码: dake

2. **设置：**
   - 下载后，请将模型文件放入`assets`文件夹。
   - 解压存放在`libs/arm64-v8a`文件夹中的`*.so`压缩文件。

3. **模型信息：**
   - 演示模型名为**FSMN-VAD**、**Paraformer**和**ERes2Net**，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
   - 因此，演示模型的输入输出与原始模型略有不同。

4. **ONNX Runtime 适配：**
   - 为了更好地适配ONNX Runtime-Android，导出时未使用动态轴。因此，导出的ONNX模型对x86_64可能不是最佳选择。
   - 我们计划在未来公开转换导出的方法。

5. **限制：**
   - 由于模型的限制，目前英文版本的表现不佳。

6. **权限：**
   - 首次使用时，您需要先授权录音权限，然后再重新启动应用程序。

7. **性能：**
   - ASR推理的耗时大约为25毫秒，说话者确认也需要25毫秒。VAD大约需要2毫秒。
   - 在演示中，我们将计算频率默认设置为16FPS（每轮60毫秒），以便让离线ASR模型接近在线流式处理模型的性能，同时保持离线模型的准确性。

8. **功能：**
   - 自由设置您的唤醒词，系统将使用模糊音调来匹配它，减少口音的影响。醒来后，默认保持活动30秒（可调）。
   - 您可以直接发出简单命令，无需唤醒。（免唤醒模式）例如：打开xxx、关闭xxx、导航到xxx、播放某人的歌曲...等
   - 如果句子中包含“和”, "还有", "然后"...等等常见的连词，系统将进行多意图判断。
   - 只需直接说出关键词，如 '添加声音' 或 '添加权限'...，系统将识别您的声音为权限。 '删除权限' 也适用同样的操作。
   - 一旦添加权限，只有声音的所有者才能修改它。系统仅识别授权声音为有效命令。
   - 不保证权限识别成功率。更多信息，请参考ERes2Net模型介绍。

9. **量化：**
   - 模型的量化方法可以在文件夹 "Do_Quantize" 中查看。
   - 现在不建议使用q4(uint4)量化方法, 因为ONNX Runtime的运算符"MatMulNBits"表现不佳。

## 其他资源

- 看更多项目: [https://dakeqq.github.io/overview/](https://dakeqq.github.io/overview/)

---
