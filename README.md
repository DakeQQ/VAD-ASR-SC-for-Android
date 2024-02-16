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
10. In speaker confirmation mode, simply say the keywords, '添加声音' or '添加权限', directly, and the system will recognize your voice as permission. The same applies to deleting permissions.
![Demo Animation](https://github.com/DakeQQ/VAD-ASR-SC-for-Android/blob/main/asr.gif?raw=true)
