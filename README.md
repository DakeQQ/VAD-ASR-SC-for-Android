# VAD-ASR-SC-for-Android
1. Demonstration of VAD + ASR + Speaker Confirmation on Android device.
2. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1ErEdY6QMyJCW0yuQR03If905IhdyAHFw?usp=drive_link
3. After downloading, place the model into the assets folder.
4. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
5. The demo models, named 'FSMN-VAD' & 'Paraformer' & 'ERes2Net', were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
6. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
7. Due to the model's limitations, the English version does not perform well currently.
8. We will make the exported method public later.
9. You have to approve the recorder permissions first, and then relaunch the app again.
10. The time cost of ASR inference is about 25ms, and speaker confirmation also takes 25ms. VAD takes about 2ms.
11. In the demo, we set the monitoring frequency to a default of 16FPS (60ms per round) to make an offline ASR model approximate the performance of an online streaming one, while maintaining the offline model's accuracy.
12. It can be set to an awakening word of your choice, and the system will use a fuzzy tone to match it (Chinese awakening words only), minimizing the effect of natural speech variations. After waking up, it defaults to staying active for 30 seconds (adjustable).
13. You can issue simple commands directly without invoking the awakening word. (wake free mode) For example: Open xxx, Close xxx, Navigate to xxx, Play someone's song...and so on.
14. If the sentence contains conjunctions such as 'and' or other common continuation words, the system will engage in multi-intent judgment.
15. Just simply say the keywords, 'adding voice' or 'adding premission'..., directly, and the system will recognize your voice as permission. The same applies to 'deleting permission'.
16. Once permission is added, only the owner of the voice can modify it. The system will only recognize the authorized sound as an effective command.
17. No guarantee for the permission's success ratio. For more information, please refer to the ERes2Net model introduction.
18. See more projects: https://dakeqq.github.io/overview/

# 语音活动检测 + 自动语音辨识 + 说话人确认 - 安卓
1. 在Android设备上进行VAD + ASR + 说话人确认的演示。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1ErEdY6QMyJCW0yuQR03If905IhdyAHFw?usp=drive_link
3. 百度: https://pan.baidu.com/s/1Si-4ebtqm2HA9omxqHCMuQ?pwd=dake 提取码: dake
4. 下载后，请将模型文件放入assets文件夹。
5. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
6. 演示模型名为'FSMN-VAD' & 'Paraformer' & 'ERes2Net'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
7. 因此，演示模型的输入输出与原始模型略有不同。
8. 由于模型的限制，目前英文版本的表现不佳。
9. 我们未来会提供转换导出的方法。
10. 首次使用时，您需要先授权录音权限，然后再重新启动应用程序。
11. ASR推理的耗时大约为25毫秒，说话者确认也需要25毫秒。VAD大约需要2毫秒。
12. 在演示中，我们将计算频率默认设置为16FPS（每轮60毫秒），以便让离线ASR模型接近在线流式处理模型的性能，同时保持离线模型的准确性。
13. 自由设置您的唤醒词，系统将使用模糊音调来匹配它，减少口音的影响。醒来后，默认保持活动30秒（可调）。
14. 您可以直接发出简单命令，无需唤醒。（免唤醒模式）例如：打开xxx、关闭xxx、导航到xxx、播放某人的歌曲...等
15. 如果句子中包含“和”, "还有", "然后"...等等常见的连词，系统将进行多意图判断。
16. 只需直接说出关键词，如 '添加声音' 或 '添加权限'...，系统将识别您的声音为权限。 '删除权限' 也适用同样的操作。
17. 一旦添加权限，只有声音的所有者才能修改它。系统仅识别授权声音为有效命令。
18. 不保证权限识别成功率。更多信息，请参考ERes2Net模型介绍。
19. 看更多項目: https://dakeqq.github.io/overview/

# 演示结果 Demo Results
1. 此GIF以每秒7帧的速度生成。因此，ASR看起来可能不够流畅。This GIF was generated at 7fps. Therefore, it may not look smooth enough.
![Demo Animation](https://github.com/DakeQQ/VAD-ASR-SC-for-Android/blob/main/asr.gif?raw=true)
