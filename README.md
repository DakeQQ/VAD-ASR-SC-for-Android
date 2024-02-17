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
10. It can be set to an awakening word of your choice, and the system will use a fuzzy tone to match it (Chinese awakening words only), minimizing the effect of natural speech variations.
11. You can issue simple commands directly without invoking the awakening word. (wake free mode)
12. If the sentence contains conjunctions such as 'and' or other continuation words, the system will engage in multi-intent judgment.
13. Just simply say the keywords, 'adding voice' or 'adding premission'..., directly, and the system will recognize your voice as permission. The same applies to 'deleting permission'.
14. Once permission is added, only the owner of the voice can modify it. The system will only recognize the authorized sound as an effective command.
15. No guarantee for the permission's success ratio. For more information, please refer to the ERes2Net model introduction.

# VAD+ASR+说话人确认-安卓
1. 在Android设备上进行VAD + ASR + 说话人确认的演示。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1ErEdY6QMyJCW0yuQR03If905IhdyAHFw?usp=drive_link
3. 下载后，请将模型文件放入assets文件夹。
4. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
5. 演示模型名为'FSMN-VAD' & 'Paraformer' & 'ERes2Net'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
6. 因此，演示模型的输入输出与原始模型略有不同。
7. 由于模型的限制，目前英文版本的表现不佳。
8. 我们未来会提供转换导出的方法。
8. 首次使用时，您需要先授权录音权限，然后再重新启动应用程序。
10. 自由设置您的唤醒词，系统将使用模糊音调来匹配它，减少口音的影响。
11. 您可以直接发出简单命令，无需唤醒。（免唤醒模式）
12. 如果句子中包含“和”, "还有", "然后"...等等连词，系统将进行多意图判断。
13. 只需直接说出关键词，如 '添加声音' 或 '添加权限'...，系统将识别您的声音为权限。 '删除权限' 也适用同样的操作。
14. 一旦添加权限，只有声音的所有者才能修改它。系统仅识别授权声音为有效命令。
15. 不保证权限识别成功率。更多信息，请参考ERes2Net模型介绍。

# 演示结果 Demo Results
![Demo Animation](https://github.com/DakeQQ/VAD-ASR-SC-for-Android/blob/main/asr.gif?raw=true)
