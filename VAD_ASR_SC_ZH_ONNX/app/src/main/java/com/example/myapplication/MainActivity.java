package com.example.myapplication;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.WindowManager;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.myapplication.databinding.ActivityMainBinding;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.py.Pinyin;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.channels.Channels;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    private static final int SAMPLE_RATE = 48000;  // The sampling rate.
    private static final int FRAME_BUFFER_SIZE_MONO_48k = 2868;  //  2868 means 60ms per loop at 48kHz. Please edit the 'loop_time', which in the project.h, at the same time. The recommends range is 60 ~ 100ms, due to the model was exported with (1, 15, 560) static shape.
    private static final int FRAME_BUFFER_SIZE_MONO_16k = FRAME_BUFFER_SIZE_MONO_48k / 3;
    private static final int FRAME_BUFFER_SIZE_STEREO = FRAME_BUFFER_SIZE_MONO_48k * 2;
    private static final int paraformer_input_shape = 560;  // The same variable in the project.h, please modify at the same time.
    private static final int vad_input_shape = 400;  // The same variable in the project.h, please modify at the same time.
    private static final int continue_hit_threshold = 1;  // 1 means: 1 words matched. 2 means 2 words matched in continuously...
    private static final int effective_awake_duration = 30000;  // unit: ms. After this time passes, the system will go into sleep mode.
    private static final int amount_of_mic = 1;  // The same variable in the project.h, please modify at the same time.///////////////////////////////////////////////////////////////////////////////////////////////////////
    private static final int amount_of_mic_channel = amount_of_mic * 1;  // number of channels per mic, The same variable in the project.h, please modify at the same time.///////////////////////////////////////////////////////////////////////////////////////////////////////
    private static final int pre_allocate_num_words = 25;  // The same variable in the project.h, please modify at the same time.
    private static final int amount_of_timers = amount_of_mic_channel;  // Timers are used to count the time in activation (wake-up).
    private static final int all_record_size = amount_of_mic_channel * FRAME_BUFFER_SIZE_MONO_48k;
    private static final int all_record_size_16k = all_record_size / 3;  // 48k to 16k
    private static final int continue_threshold = 3;  // Set a continuous speaking threshold to avoid capturing undesired speech.
    private static final int asr_temp_save_limit = 50;
    private static final int print_threshold = 3;  // The waiting loop counts before sending the ASR results. Sometimes, the VAD is too sensitive/insensitive and cuts off speech prematurely.
    private static final int model_hidden_size_Res2Net = 512;
    private static int sub_commands = 0;
    private static int reduplication_words_count = 0;
    private static int temp_stop = -1;
    private static int awake_id = -1;
    private static int amount_of_speakers = 0; // The speakers who have permission are stored in the speaker_features.txt file.
    static final int font_size = 18;
    private static final float threshold_Speaker_Confirm = 0.36f;  //  You can print the max_score, which was calculated in the Compare_Similarity, to assign a appropriate value.
    private static final int[] continue_active = new int[amount_of_mic_channel];
    private static final int[] print_count = new int[amount_of_mic_channel];
    private static final float[] negMean_asr = new float[paraformer_input_shape];
    private static final float[] invStd_asr = new float[paraformer_input_shape];
    private static final float[] negMean_vad = new float[vad_input_shape];
    private static final float[] invStd_vad = new float[vad_input_shape];
    private static final float[][] score_data_Speaker = new float[5][model_hidden_size_Res2Net];  // Pre-allocate the storage space for speaker confirm.
    private static final float[] score_pre_calculate_Speaker = new float[score_data_Speaker.length];
    private static final String arousal_word_default = "你好大可";  // If use English words, lower case only. Due to the limitations of the open-source model, the English ASR is not performing well at the moment.
    private static final String file_name_vocab_asr = "vocab_asr.txt";
    private static final String file_name_negMean_asr = "negMean_asr.txt";
    private static final String file_name_invStd_asr = "invStd_asr.txt";
    private static final String file_name_negMean_vad = "negMean_vad.txt";
    private static final String file_name_invStd_vad = "invStd_vad.txt";
    private static final String file_name_response = "wake_up_response.txt";
    private static final String file_name_speakers = "speaker_features.txt";
    @SuppressLint("SdCardPath")
    private static final String cache_path = "/data/user/0/com.example.myapplication/cache/";
    private static final String asr_turn_on = "启动ASR \nASR Starting";
    private static final String enter_question = "请输入问题 \nEnter Questions";
    private static final String cleared = "已清除 Cleared";
    private static final String restart_success = "已重新启动 Restarted";
    private static final String restart_failed = "重新启动失败, 请退出此应用程序并手动重新启动。\nRestart Failed. Please exit this App and manually restart.";
    private static final String exit_wake_up = "没啥事我就先睡了, 有事喊我。\nTimes up, please wake me up again.";
    private static final String command_queue_state_zh = "个命令正在等待执行。";
    private static final String command_queue_state_en = " commands are waiting for execute.";
    private static final String set_arousal_words = "已设置唤醒词为The arousal word has been set as:\n";
    private static final String voice_existed = "\n此声音权限已经存在。\nThe speaker's permission has already added.";
    private static final String voice_added = "\n已添加此声音权限。\nThe speaker's permission has been added.";
    private static final String voice_full = "声音权限的存储空间已满。\nIt is out of storage space.";
    private static final String voice_deleted = "\n已删除此声音权限。\nThe speaker's permission has been deleted.";
    private static final String voice_unknown = "此声音权限以前未添加过。\nThe speaker's permission has not been added before.";
    private static final String add_permission = "添加声纹权限\nAdd the voice permission.";
    private static final String delete_permission = "删除声纹权限\nDelete the voice permission.";
    private static final String[] speech2text = new String[amount_of_mic_channel];
    private static final String[] mic_owners = {"主驾驶-Master", "副驾驶-Co_Pilot", "左后座-Lefter", "右后座-Righter"};  // The corresponding name of the mic. This size must >= amount_of_mic.
    private static final String[] add_voice_permission = {"添加声音", "添加权限", "添加限制", "加入权限", "加入声纹"}; // The key words for add the permission of voice control.
    private static final String[] delete_voice_permission = {"删除声音", "删掉声音", "删除权限", "删掉权限", "移出权限", "移除限制"};  // The key words for delete the permission of voice control.
    private static final String[] and_words = {"和", "跟", "还有", "然后", "还要", "再", "后", "接着", "加上", "之后", "以及", "最后", "还想", "与"};  // This set is used to split multi-intention tasks.
    private static final String[] open_words = {"打开", "开", "开启", "启动"};   // This set is used to continue the previous tasks' intentions. Foe example: "打开窗户和空调" -> "1.打开窗户; 2.开启空调;"
    private static final String[] close_words = {"关", "关掉", "关闭", "关上", "切", "切掉", "停掉", "停止", "切断", "闭嘴", "结束"};  // This set is used to continue the previous tasks' intentions.
    private static final String[] first_key_words = {"关", "打", "开", "开", "启", "关", "关", "切", "切", "停", "停", "切", "闭", "结", "播", "播", "导", "温", "风", "风", "音", "声", "上", "下"};
    // first_key_words & second_key_words are used to match common commands without wake-up. Further commands require activation.
    private static final String[] second_key_words = {" ", "开", "启", " ", "动", "闭", "上", " ", "掉", "掉", "止", "断", "嘴", "束", " ", "放", "航", "度", "量", "速", "量", "音", "一", "一"};
    private static String usrInputText = "";
    private static String arousal_word = arousal_word_default;
    private static String[] pre_speech2text = new String[1];
    private static final StringBuilder asr_string_builder = new StringBuilder();
    private static List<Pinyin> arousal_pinyinList;
    private static final List<Integer> user_queue = new ArrayList<>();
    private static final List<String> wake_up_response = new ArrayList<>();
    private static final List<String> command_queue = new ArrayList<>();
    private static final List<String> command_history = new ArrayList<>();
    private static final List<String> vocab_asr = new ArrayList<>();
    private static final List<List<String>> asr_record = new ArrayList<>();
    private static final List<List<Integer>> asr_permission = new ArrayList<>();
    private static final boolean focus_mode = false;  // If true, the ASR only processes the awake_id queries until de-activate. If false, the ASR processes all mic queries, which means it can be activated and receive commands from different users at will.
    private static boolean awake_response = false;
    private static boolean strict_mode = false;
    private static boolean reduplication_arousal_words = false;
    private static boolean end_of_answer = true;
    private static boolean got_key_words = false;
    private static boolean english_arousal_words = false;  // Due to the arousal_word_default was set as '你好大可'.
    private static final boolean[] arousal_awake = new boolean[amount_of_mic_channel];
    private static final boolean[] save_record = new boolean[amount_of_mic_channel];
    Button clearButton;
    Button sendButton;
    Button restartButton;
    Button renewButton;
    ImageView set_photo;
    @SuppressLint("StaticFieldLeak")
    static AutoCompleteTextView inputBox;
    static RecyclerView answerView;
    private static ChatAdapter chatAdapter;
    private static List<ChatMessage> messages;
    private static ASRThread asrThread;
    private static final Random random = new Random();
    private static final TimerManager timerManager = new TimerManager(amount_of_timers);
    private static MultiMicRecorder multiMicRecorder;

    static {
        System.loadLibrary("myapplication");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        com.example.myapplication.databinding.ActivityMainBinding binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
        AssetManager mgr = getAssets();
        runOnUiThread(() -> Load_Models_0(mgr,false,false,false,false,false,false)); // Actually, there are no options that you can set to 'true' to achieve better performance than with ONNX Runtime itself.
        runOnUiThread(() -> Load_Models_1(mgr,false,false,false,false,false,false)); // Therefore, we do not use an if-else statement to determine whether the loading was successful or not.
        runOnUiThread(() -> Load_Models_2(mgr,false,false,false,false,false,false));
        runOnUiThread(() -> {
            Read_Assets(file_name_vocab_asr, mgr);
            Read_Assets(file_name_negMean_asr, mgr);
            Read_Assets(file_name_invStd_asr, mgr);
            Read_Assets(file_name_negMean_vad, mgr);
            Read_Assets(file_name_invStd_vad, mgr);
            Read_Assets(file_name_response, mgr);
            Pre_Process(negMean_asr, invStd_asr, negMean_vad, invStd_vad);
        });
        runOnUiThread(() -> {
            for (float[] data : score_data_Speaker) {
                data[0] = -999.f;
            }
            Read_Assets(file_name_speakers, mgr);
            for (int i = 0; i < score_pre_calculate_Speaker.length; i++) {
                if (score_data_Speaker[i][0] != -999.f) {
                    score_pre_calculate_Speaker[i] = (float) Math.sqrt(Dot(score_data_Speaker[i], score_data_Speaker[i]));
                    amount_of_speakers += 1;
                } else {
                    score_pre_calculate_Speaker[i] = 1.f;
                }
            }
        });
        arousal_pinyinList = HanLP.convertToPinyinList(arousal_word_default);
        showToast(set_arousal_words + arousal_word_default,true);
        set_photo = findViewById(R.id.role_image);
        clearButton = findViewById(R.id.clear);
        sendButton = findViewById(R.id.send);
        renewButton = findViewById(R.id.renew);
        restartButton = findViewById(R.id.restart);
        inputBox = findViewById(R.id.input_text);
        messages = new ArrayList<>();
        chatAdapter = new ChatAdapter(messages);
        answerView = findViewById(R.id.result_text);
        answerView.setLayoutManager(new LinearLayoutManager(this));
        answerView.setAdapter(chatAdapter);
        set_photo.setImageResource(R.drawable.psyduck);
        for (int i = 0; i < amount_of_mic_channel; i++) {
            List<String> temp = new ArrayList<>();
            List<Integer> temp2 = new ArrayList<>();
            asr_record.add(temp);
            asr_permission.add(temp2);
            arousal_awake[i] = false;
            speech2text[i] = "";
            print_count[i] = 0;
        }
        Reset_Arousal_Words();
        clearButton.setOnClickListener(v -> clearHistory());
        restartButton.setOnClickListener(v -> Restart());
        myInit();
        Init_Chat();
        startASR();
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
        showToast(asr_turn_on, false);
    }
    private class ASRThread extends Thread {
        private boolean recording = true;
        @Override
        public void run() {
            while (recording) {
                float[] recordedData = multiMicRecorder.Read_PCM_Data();
                runOnUiThread(() -> {
                    long start_time = System.currentTimeMillis();
                    int[] result = Run_VAD_ASR(FRAME_BUFFER_SIZE_MONO_16k, recordedData, arousal_awake, focus_mode, temp_stop);
//                    System.out.println("ASR_Time_Cost: " + (System.currentTimeMillis() - start_time) + "ms");
                    int index_i = 0;
                    for (int i = 0; i < amount_of_mic_channel; i++) {
                        if (result[index_i] != -1) {
                            int permission_check = 0;
                            if (amount_of_speakers > 0) {
                                if (Compare_Similarity(i) != -1) {
                                    permission_check = 1;
                                } else {
                                    permission_check = -1;
                                }
                            }
                            continue_active[i] += 1;
                            print_count[i] = 0;
                            for (int j = 0; j < pre_allocate_num_words; j++) {
                                if (result[index_i + j] != -1) {
                                    asr_string_builder.append(vocab_asr.get(result[index_i + j]));
                                } else {
                                    break;
                                }
                            }
                            speech2text[i] = asr_string_builder.toString().replace("@", "");
                            asr_string_builder.setLength(0);
                            System.out.println("mic_" + mic_owners[i] + ": " + speech2text[i]); /////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            boolean add_voice = false;
                            boolean delete_voice = false;
                            for (String s : add_voice_permission) {
                                if (speech2text[i].equals(s)) {
                                    add_voice = true;
                                    break;
                                }
                            }
                            if (!add_voice) {
                                for (String s : delete_voice_permission) {
                                    if (speech2text[i].equals(s)) {
                                        delete_voice = true;
                                        break;
                                    }
                                }
                            }
                            if (add_voice) {
                                addHistory(ChatMessage.TYPE_USER, add_permission);
                                if (amount_of_speakers < score_pre_calculate_Speaker.length) {
                                    int speaker = Compare_Similarity(i);
                                    if (speaker != -1) {
                                        addHistory(ChatMessage.TYPE_SYSTEM,"Speaker_ID: " + speaker + voice_existed);
                                    } else {
                                        for (int j = 0; j < score_pre_calculate_Speaker.length; j++) {
                                            if (score_data_Speaker[j][0] == -999.f) {
                                                addHistory(ChatMessage.TYPE_SYSTEM,"Speaker_ID: " + j + voice_added);
                                                score_data_Speaker[j] = Run_Speaker_Confirm(i, model_hidden_size_Res2Net);
                                                score_pre_calculate_Speaker[j] = (float) Math.sqrt(Dot(score_data_Speaker[j], score_data_Speaker[j]));
                                                saveToFile(score_data_Speaker, cache_path + file_name_speakers);
                                                amount_of_speakers += 1;
                                                break;
                                            }
                                        }
                                    }
                                } else {
                                    addHistory(ChatMessage.TYPE_SYSTEM, voice_full);
                                }
                                temp_stop = i;
                            } else if (delete_voice) {  // Only the speaker's own voice is removed, as no ID extraction methods are applied.
                                addHistory(ChatMessage.TYPE_USER, delete_permission);
                                int speaker = Compare_Similarity(i);
                                if (speaker != -1) {
                                    addHistory(ChatMessage.TYPE_SYSTEM, "Speaker_ID: " + speaker + voice_deleted);
                                    score_data_Speaker[speaker][0] = -999.f;
                                    score_pre_calculate_Speaker[speaker] = 1.f;
                                    saveToFile(score_data_Speaker, cache_path + file_name_speakers);
                                    amount_of_speakers -= 1;
                                } else {
                                    addHistory(ChatMessage.TYPE_SYSTEM, voice_unknown);
                                }
                                temp_stop = i;
                            } else {
                                String[] check_usrInputText = speech2text[i].split("");
                                String[] inverse_check_usrInputText = new String[check_usrInputText.length];
                                int n = 0;
                                for (int j = check_usrInputText.length - 1; j >= 0; j--) {
                                    inverse_check_usrInputText[n] = check_usrInputText[j];
                                    n++;
                                }
                                int compare_count = 0;
                                for (int j = 0; j < check_usrInputText.length; j++) {  // Only compare the adjacent words
                                    for (int k = j - 1; k <= j + 1; k++) {
                                        if (k < 0) {
                                            continue;
                                        } else if (k >= pre_speech2text.length) {
                                            break;
                                        }
                                        if (Objects.equals(inverse_check_usrInputText[j], pre_speech2text[k])) {
                                            compare_count += 1;
                                            if (compare_count >= check_usrInputText.length) {
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (compare_count >= check_usrInputText.length) {
                                    if (Objects.equals(inverse_check_usrInputText[0], pre_speech2text[0])) {
                                        if (pre_speech2text.length >= check_usrInputText.length) {
                                            print_count[i] += 1;
                                            if (print_count[i] >= print_threshold) {
                                                save_record[i] = false;
                                                temp_stop = i;
                                            }
                                        } else {
                                            print_count[i] = 1;
                                        }
                                    }
                                } else {
                                    print_count[i] = 0;
                                }
                                pre_speech2text = inverse_check_usrInputText;
                                if (save_record[i]) {
                                    asr_record.get(i).add(speech2text[i]);
                                    asr_permission.get(i).add(permission_check);
                                    if (asr_record.get(i).size() > asr_temp_save_limit) {
                                        asr_record.get(i).remove(0);
                                        asr_permission.get(i).remove(0);
                                    }
                                }
                            }
                        } else {
                            temp_stop = -1;
                            if (print_count[i] <= print_threshold) {
                                print_count[i] += 1;
                            } else {
                                print_count[i] = print_threshold;
                                if (asr_record.size() > 0) {
                                    asr_record.get(i).clear();
                                    asr_permission.get(i).clear();
                                }
                            }
                        }
                        index_i += pre_allocate_num_words;
                    }
                    for (int k = 0; k < amount_of_mic_channel; k++) {
                        if ((!Objects.equals(speech2text[k], "")) && (continue_active[k] > continue_threshold)) {
                            int permission_gate = 0;
                            if (amount_of_speakers > 0) {
                                for (int i = 0; i < asr_permission.get(k).size(); i++) {
                                    permission_gate += asr_permission.get(k).get(i);
                                }
                            }
                            if (permission_gate < 0) {
                                addHistory(ChatMessage.TYPE_SYSTEM, "对不起，您没有权限。\nSorry, you don't have the permission.");
                                speech2text[k] = "";
                                pre_speech2text = new String[0];
                                asr_record.get(k).clear();
                                asr_permission.get(k).clear();
                                save_record[k] = true;
                                continue_active[k] = 0;
                                continue;
                            }
                            if (!arousal_awake[k]) {
                                int compare_count = 0;
                                if (english_arousal_words) {
                                    String[] asr_result = speech2text[k].split(" ");
                                    String[] arousal_answer = arousal_word.split(" ");
                                    for (int i = 0; i < arousal_answer.length; i++) {  // Only compare the adjacent words
                                        for (int j = i - 1; j <= i + 1; j++) {
                                            if (j < 0) {
                                                continue;
                                            } else if (j >= asr_result.length) {
                                                break;
                                            }
                                            if (Objects.equals(asr_result[j], arousal_answer[i])) {
                                                compare_count += 1;
                                            }
                                        }
                                    }
                                    if (arousal_answer.length < 2) {  // Strict judge mode
                                        if (compare_count == arousal_answer.length && (asr_result.length == arousal_answer.length)) {
                                            arousal_awake[k] = true;
                                            awake_response = true;
                                            speech2text[k] = arousal_word;
                                        } else {
                                            arousal_awake[k] = false;
                                        }
                                    } else if ((Math.abs(compare_count - arousal_answer.length) < 2) && (Math.abs(asr_result.length - arousal_answer.length) < 2)) {
                                        arousal_awake[k] = true;
                                        awake_response = true;
                                        speech2text[k] = arousal_word;
                                    } else {
                                        arousal_awake[k] = false;
                                    }
                                } else {
                                    List<Pinyin> asr_pinyinList = HanLP.convertToPinyinList(speech2text[k]);
                                    if (reduplication_arousal_words) {
                                        compare_count -= reduplication_words_count;
                                    }
                                    for (int i = 0; i < arousal_pinyinList.size(); i++) {  // Only compare the adjacent words
                                        for (int j = i - 1; j <= i + 1; j++) {
                                            if (j < 0) {
                                                continue;
                                            } else if (j >= asr_pinyinList.size()) {
                                                break;
                                            }
                                            if (Objects.equals(asr_pinyinList.get(j).getPinyinWithoutTone(), arousal_pinyinList.get(i).getPinyinWithoutTone())) {
                                                compare_count += 1;
                                            }
                                        }
                                    }
                                    if (strict_mode) {
                                        if (compare_count == arousal_pinyinList.size() && (speech2text[k].length() == arousal_pinyinList.size())) {
                                            arousal_awake[k] = true;
                                            awake_response = true;
                                            speech2text[k] = arousal_word;
                                        } else {
                                            arousal_awake[k] = false;
                                        }
                                    } else if ((Math.abs(compare_count - arousal_pinyinList.size()) < 2) && (Math.abs(speech2text[k].length() - arousal_pinyinList.size()) < 2)) {
                                        arousal_awake[k] = true;
                                        awake_response = true;
                                        speech2text[k] = arousal_word;
                                    } else {
                                        arousal_awake[k] = false;
                                    }
                                }
                            }
                            List<String> check_key_words = Match_Key_Words(asr_record.get(k));
                            if (arousal_awake[k] | got_key_words) {
                                if (end_of_answer) {
                                    addHistory(ChatMessage.TYPE_USER, speech2text[k]);
                                }
                                if (awake_response) {
                                    int finalK1 = k;
                                    runOnUiThread(() -> {
                                        if (end_of_answer) {
                                            addHistory(ChatMessage.TYPE_SERVER, "Response to:\n回复" + mic_owners[finalK1] + ": " + wake_up_response.get(random.nextInt(wake_up_response.size())));
                                        } else {
                                            showToast("接收来自" + mic_owners[finalK1] + "的唤醒。" + "\nNew wake up by " + mic_owners[finalK1],false);
                                        }
                                    });
                                    temp_stop = k;
                                    speech2text[k] = "";
                                    pre_speech2text = new String[0];
                                    asr_record.get(k).clear();
                                    asr_permission.get(k).clear();
                                    save_record[k] = true;
                                    awake_response = false;
                                    continue_active[k] = 0;
                                    continue;
                                }
                                if (arousal_awake[k]) {
                                    timerManager.resetTimer(k);
                                    timerManager.startTimer_awake(k);
                                }
                                if (print_count[k] >= print_threshold) {
                                    temp_stop = k;
                                    speech2text[k] = "";
                                    pre_speech2text = new String[0];
                                    save_record[k] = false;
                                    continue_active[k] = 0;
                                    for (int i = continue_threshold; i > 0; i--) {
                                        if (check_key_words.size() > i) {
                                            check_key_words.subList(0, i).clear();
                                            break;
                                        }
                                    }
                                    int finalK = k;
                                    runOnUiThread(() -> {
                                        usrInputText = Sentence_Stitching(check_key_words, continue_hit_threshold);
                                        user_queue.add(finalK);
                                        asr_record.get(finalK).clear();
                                        asr_permission.get(finalK).clear();
                                        command_queue.add(usrInputText);
                                        save_record[finalK] = true;
                                    });
                                }
                            } else {
                                speech2text[k] = "";
                                pre_speech2text = new String[0];
                                asr_record.get(k).clear();
                                asr_permission.get(k).clear();
                                save_record[k] = true;
                                continue_active[k] = 0;
                            }
                        }
                    }
                });
                runOnUiThread(MainActivity.this::Run_Command_Queue);
            }
        }
        private void stopASR() {
            recording = false;
            awake_response = false;
            multiMicRecorder.stopRecording();
            asr_string_builder.setLength(0);
            pre_speech2text = new String[0];
            for (int k = 0; k < amount_of_mic_channel; k++) {
                speech2text[k] = "";
                arousal_awake[k] = false;
                asr_record.get(k).clear();
                asr_permission.get(k).clear();
                print_count[k] = 0;
            }
            for (int k = 0; k < amount_of_timers; k++) {
                timerManager.resetTimer(k);
            }
        }
    }
    private static class TimerManager {
        private Timer[] timers;
        private TimerManager(int numberOfTimers) {
            createTimers(numberOfTimers);
        }
        private void createTimers(int numberOfTimers) {
            timers = new Timer[numberOfTimers];
            for (int i = 0; i < numberOfTimers; i++) {
                timers[i] = new Timer();
            }
        }
        private void startTimer_awake(int timerID) {
            TimerTask task = new TimerTask() {
                @Override
                public void run() {
                    arousal_awake[timerID] = false;
                    new Handler(Looper.getMainLooper()).post(() -> addHistory(ChatMessage.TYPE_SYSTEM, exit_wake_up));
                    timers[timerID].cancel();
                    timers[timerID] = new Timer();
                }
            };
            timers[timerID].schedule(task, MainActivity.effective_awake_duration);
        }
        private void resetTimer(int timerID) {
            timers[timerID].cancel();
            timers[timerID] = new Timer();
        }
    }
    private class MultiMicRecorder {
        private final AudioRecord[] audioRecords;
        private final short[][] bufferArrays;
        private final Thread[] recordThreads;
        private MultiMicRecorder() {
            audioRecords = new AudioRecord[amount_of_mic];
            bufferArrays = new short[amount_of_mic][FRAME_BUFFER_SIZE_STEREO];
            recordThreads = new Thread[amount_of_mic];
            if (ContextCompat.checkSelfPermission(MainActivity.this, android.Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{android.Manifest.permission.RECORD_AUDIO},1);
            } else {
                for (int i = 0; i < amount_of_mic; i++) {
                    audioRecords[i] = new AudioRecord(MediaRecorder.AudioSource.VOICE_PERFORMANCE, SAMPLE_RATE,
                            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, FRAME_BUFFER_SIZE_STEREO);  // One microphone usually has two channels: left & right. You can use CHANNEL_IN_STEREO instead of CHANNEL_IN_MONO, and then separate the stereo recording results into odd & even indices to obtain the left & right PCM data, respectively.
                    final int micIndex = i;
                    recordThreads[i] = new Thread(() -> {
                        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);
                        audioRecords[micIndex].startRecording();
                    });
                }
            }
        }
        private void startRecording() {
            for (Thread thread : recordThreads) {
                thread.start();
            }
        }
        private void stopRecording() {
            for (AudioRecord audioRecord : audioRecords) {
                audioRecord.stop();
                audioRecord.release();
            }
        }
        private float[] Read_PCM_Data() {
            short[] resultArray = new short[all_record_size];
            Thread[] readThreads = new Thread[amount_of_mic];
            for (int i = 0; i < amount_of_mic; i++) {
                final int micIndex = i;
                readThreads[i] = new Thread(() -> {
                    int bytesRead = audioRecords[micIndex].read(bufferArrays[micIndex], 0, FRAME_BUFFER_SIZE_MONO_48k);  // If use STEREO, FRAME_BUFFER_SIZE_STEREO instead.
                    System.arraycopy(bufferArrays[micIndex], 0, resultArray, micIndex * FRAME_BUFFER_SIZE_MONO_48k, bytesRead); // If use STEREO, FRAME_BUFFER_SIZE_STEREO instead.
                });
            }
            for (Thread thread : readThreads) {
                thread.start();
            }
            try {
                for (Thread thread : readThreads) {
                    thread.join();
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
////    If you would like using STEREO recording, please open the following code.
////    Remember to edit the "amount_of_mic_channel" both in project.h & Java. Also the "AudioFormat.CHANNEL_IN_MONO" -> "AudioFormat.CHANNEL_IN_STEREO", "bytesRead: FRAME_BUFFER_SIZE_MONO_48k -> FRAME_BUFFER_SIZE_STEREO"
//            {
//                short[] odd = new short[FRAME_BUFFER_SIZE_MONO_48k * amount_of_mic];
//                short[] even = new short[odd.length];
//                int micIndex = 0;
//                for (int i = 0; i < amount_of_mic; i++) {
//                    int count = 0;
//                    for (int j = 0; j < FRAME_BUFFER_SIZE_STEREO; j += 2) {
//                        even[count] = resultArray[micIndex + j];
//                        odd[count] = resultArray[micIndex + j + 1];
//                        count += 1;
//                    }
//                    System.arraycopy(even, 0, resultArray, micIndex, FRAME_BUFFER_SIZE_MONO_48k);
//                    System.arraycopy(odd, 0, resultArray, micIndex + FRAME_BUFFER_SIZE_MONO_48k, FRAME_BUFFER_SIZE_MONO_48k);
//                    micIndex += FRAME_BUFFER_SIZE_STEREO;
//                }
//            }
            float[] all_mic_record = new float[all_record_size_16k];  // Down sampling from 48kHz to 16kHz to achieve greater accuracy. Yor can use 16kHz float32 PCM directly without the following process.
            int index = 0;
            for (int i = 1; i < all_record_size - 1; i+=3) {
                all_mic_record[index] = (resultArray[i - 1] + resultArray[i] + resultArray[i] + resultArray[i + 1]) * 0.25f;  // Central weight average.
                index += 1;
            }
            return all_mic_record;
        }
    }
    private void Reset_Arousal_Words() {
        renewButton.setOnClickListener(view -> {
            arousal_word = String.valueOf(inputBox.getText());
            inputBox.setText("");
            if (Objects.equals(arousal_word, "")) {
                arousal_word = arousal_word_default;
                english_arousal_words = false;
            }
            if (arousal_word.split("")[0].matches("^[A-Za-z]+$")) {
                english_arousal_words = true;
                arousal_word = arousal_word.toLowerCase();
            } else {
                english_arousal_words = false;
                arousal_pinyinList = HanLP.convertToPinyinList(arousal_word);  // Fuzzy tone matched for chinese arousal_word.
                strict_mode = arousal_pinyinList.size() < 3;
                reduplication_arousal_words = false;
                reduplication_words_count = 0;
                for (int i = 1; i < arousal_pinyinList.size(); i++) {
                    if (Objects.equals(arousal_pinyinList.get(i).getPinyinWithoutTone(), arousal_pinyinList.get(i - 1).getPinyinWithoutTone())) {
                        reduplication_arousal_words = true;
                        reduplication_words_count += 1;
                    }
                }
                if (reduplication_arousal_words) {
                    reduplication_words_count += 1;
                }
            }
            showToast("已设置唤醒词为: " + arousal_word + "\nThe arousal word has been set as: " + arousal_word, true);
        });
    }
    private void Init_Chat() {
        sendButton.setOnClickListener(view -> {
            if (usrInputText.isEmpty()) {
                usrInputText = String.valueOf(inputBox.getText());
                inputBox.setText("");
                if (usrInputText.isEmpty()){
                    showToast(enter_question,false);
                    myInit();
                    return;
                }
            }
            if (sub_commands < 1) {
                runOnUiThread(() -> addHistory(ChatMessage.TYPE_USER, usrInputText));
            }
            for (String word : and_words) {
                String temp_usrInputText = usrInputText.replace(word, ",");
                if (!temp_usrInputText.equals(usrInputText)) {
                    usrInputText = temp_usrInputText;
                    break;
                }
            }
            String[] usrInputText_array = usrInputText.split(",");
            {
                String intent_action;
                boolean on_check = false;
                boolean off_check = false;
                for (String word : open_words) {
                    String temp_first_command_on = usrInputText.replace(word, ",");
                    if (!temp_first_command_on.equals(usrInputText)) {
                        on_check = true;
                        break;
                    }
                }
                for (String word : close_words) {
                    String temp_first_command_off = usrInputText.replace(word, ",");
                    if (!temp_first_command_off.equals(usrInputText)) {
                        off_check = true;
                        break;
                    }
                }
                if (on_check && !off_check) {
                    intent_action = "开启";
                } else if (!on_check && off_check) {
                    intent_action = "关闭";
                } else {
                    intent_action = "";
                }
                for (int i = 0; i < usrInputText_array.length; i++) {
                    if (!Objects.equals(usrInputText_array[i], "")) {
                        usrInputText = usrInputText_array[i];
                        boolean sub_commands_add = false;
                        if (sub_commands < 1) {
                            sub_commands += 1;
                            sub_commands_add = true;
                        }
                        for (int j = i + 1; j < usrInputText_array.length; j++) {
                            if (!Objects.equals(usrInputText_array[j], "")) {
                                command_queue.add(0, intent_action + usrInputText_array[j]);
                                user_queue.add(0, awake_id);
                                if (sub_commands_add) {
                                    sub_commands += 1;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            sub_commands -= 1;
            addHistory(ChatMessage.TYPE_SERVER,"\nUser: " + mic_owners[awake_id] + "\nRun the command: " + "\n" + usrInputText);
            myInit();
        });
    }
    @SuppressLint("SetTextI18n")
    private void startASR() {
        multiMicRecorder = new MultiMicRecorder();
        multiMicRecorder.startRecording();
        asrThread = new ASRThread();
        asrThread.start();
    }
    @SuppressLint("SetTextI18n")
    private void stopASR() {asrThread.stopASR();}
    @SuppressLint("NotifyDataSetChanged")
    private static void addHistory(int messageType, String result) {
        int lastMessageIndex = messages.size() - 1;
        if (messageType == ChatMessage.TYPE_SYSTEM) {
            messages.add(new ChatMessage(messageType, result));
        } else if (lastMessageIndex >= 0 && messages.get(lastMessageIndex).type() == messageType) {
            if (messageType != ChatMessage.TYPE_USER ) {
                messages.set(lastMessageIndex, new ChatMessage(messageType, messages.get(lastMessageIndex).content() + result));
                for (int i = 0; i < amount_of_mic_channel; i++) {
                    if (arousal_awake[i]) {
                        timerManager.resetTimer(i);
                        timerManager.startTimer_awake(i);
                    }
                }
            } else {
                messages.set(lastMessageIndex, new ChatMessage(messageType, result));
            }
        } else {
            messages.add(new ChatMessage(messageType, result));
        }
        chatAdapter.notifyDataSetChanged();
        answerView.smoothScrollToPosition(messages.size() - 1);
    }
    @SuppressLint("NotifyDataSetChanged")
    private void clearHistory(){
        command_queue.clear();
        command_history.clear();
        user_queue.clear();
        inputBox.setText("");
        messages.clear();
        chatAdapter.notifyDataSetChanged();
        answerView.smoothScrollToPosition(0);
        myInit();
        asr_string_builder.setLength(0);
        for (int k = 0; k < amount_of_mic_channel; k++) {
            speech2text[k] = "";
            arousal_awake[k] = false;
            save_record[k] = true;
            asr_record.get(k).clear();
            asr_permission.get(k).clear();
            print_count[k] = 0;
        }
        for (int k = 0; k < amount_of_timers; k++) {
            timerManager.resetTimer(k);
        }
        pre_speech2text = new String[0];
        awake_response = false;
        sub_commands = 0;
        showToast( cleared,false);
    }
    @SuppressLint("NotifyDataSetChanged")
    private void Restart(){
        try {
            stopASR();
            startASR();
            clearHistory();
            arousal_word = arousal_word_default;
            strict_mode = false;
            reduplication_arousal_words = false;
            reduplication_words_count = 0;
            Init_Chat();
            showToast( restart_success,false);
        } catch (Exception e) {
            showToast(restart_failed,false);
        }
    }
    private static void myInit() {
        usrInputText = "";
        end_of_answer = true;
        temp_stop = -1;
        awake_id = -1;
    }
    private void Run_Command_Queue() {
        if (end_of_answer & command_queue.size() > 0) {
            end_of_answer = false;
            usrInputText = command_queue.get(0);
            command_history.add(usrInputText);  // Temporary not using in this demo.
            awake_id = user_queue.get(0);
            user_queue.remove(0);
            command_queue.remove(0);
            if (command_queue.size() > 1) {
                showToast(command_queue.size() + command_queue_state_zh + "\n" + command_queue.size() + command_queue_state_en,false);
            }
            if (command_history.size() > asr_temp_save_limit) {
                command_history.remove(0);
            }
            sendButton.performClick();
        }
    }
    private void showToast(final String content, boolean display_long){
        if (display_long) {
            Toast.makeText(this, content, Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(this, content, Toast.LENGTH_SHORT).show();
        }
    }
    private static float Dot(float[] vector1, float[] vector2) {
        float sum = 0.f;
        for (int i = 0; i < model_hidden_size_Res2Net; i++) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }
    private static int Compare_Similarity(int mic_id) {
        float[] model_result = Run_Speaker_Confirm(mic_id, model_hidden_size_Res2Net);
        float model_result_dot = (float) Math.sqrt(Dot(model_result, model_result));
        float max_score = -999.f;
        int max_position = -1;
        for (int i = 0; i < score_pre_calculate_Speaker.length; i++) {
            if (score_data_Speaker[i][0] != -999.f) {
                float temp = Dot(score_data_Speaker[i], model_result) / (score_pre_calculate_Speaker[i] * model_result_dot);
                if (temp > max_score) {
                    max_score = temp;
                    max_position = i;
                }
            }
        }
        if (max_score > threshold_Speaker_Confirm) {
            System.out.println("Speaker_ID: " + max_position + " / max_score: " + max_score);
            return max_position;
        } else {
            System.out.println("Speaker_ID: Unknown"  + " / max_score: " + max_score);
            return -1;
        }
    }
    private static String Sentence_Stitching(List<String> input_string, int continue_hit_threshold) {
        if (input_string.size() > 1) {
            boolean is_english_words = false;
            String[] s1_array = input_string.get(0).split("");
            if (s1_array[0].matches("^[A-Za-z]+$")) {
                s1_array = input_string.get(0).split(" ");
                is_english_words = true;
            }
            for (int i = 1; i < input_string.size(); i++) {
                int target_position_1 = -1;
                int target_position_2 = -1;
                String[] s2_array;
                if (is_english_words) {
                    s2_array = input_string.get(i).split(" ");
                } else {
                    s2_array = input_string.get(i).split("");
                }
                for (int j = 0; j < s1_array.length - 1; j++) {
                    for (int k = 0; k < s2_array.length - 1; k++) {
                        if (Objects.equals(s1_array[j], s2_array[k])) {
                            switch (continue_hit_threshold) {
                                case 1 -> {
                                    target_position_1 = j;
                                    target_position_2 = k;
                                    j = s1_array.length;
                                    k = s2_array.length;
                                }
                                case 2 -> {
                                    if (Objects.equals(s1_array[j + 1], s2_array[k + 1])) {
                                        target_position_1 = j;
                                        target_position_2 = k;
                                        j = s1_array.length;
                                        k = s2_array.length;
                                    }
                                }
                                case 3 -> {
                                    if (Objects.equals(s1_array[j + 1], s2_array[k + 1])) {
                                        if ((j + 2 < s1_array.length) && (k + 2 < s2_array.length)) {
                                            if (Objects.equals(s1_array[j + 2], s2_array[k + 2])) {
                                                target_position_1 = j;
                                                target_position_2 = k;
                                                j = s1_array.length;
                                                k = s2_array.length;
                                            }
                                        }
                                    }
                                }
                                case 4 -> {
                                    if (Objects.equals(s1_array[j + 1], s2_array[k + 1])) {
                                        if ((j + 2 < s1_array.length) && (k + 2 < s2_array.length)) {
                                            if (Objects.equals(s1_array[j + 2], s2_array[k + 2])) {
                                                if ((j + 3 < s1_array.length) && (k + 3 < s2_array.length)) {
                                                    if (Objects.equals(s1_array[j + 3], s2_array[k + 3])) {
                                                        target_position_1 = j;
                                                        target_position_2 = k;
                                                        j = s1_array.length;
                                                        k = s2_array.length;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (target_position_1 != -1) {
                    String[] s3_array = new String[target_position_1 + s2_array.length - target_position_2];
                    System.arraycopy(s1_array, 0, s3_array, 0, target_position_1);
                    System.arraycopy(s2_array, target_position_2, s3_array, target_position_1, s2_array.length - target_position_2);
                    s1_array = s3_array;
                } else {
                    if (s1_array.length < 2) {
                        s1_array = s2_array;
                    } else {
                        i += 1;
                        if (i >= input_string.size()) {
                            break;
                        }
                    }
                }
            }
            if (is_english_words) {
                return Arrays.toString(s1_array).replaceAll("[,\\[\\]]", "");  // Remain white space.
            } else {
                return Arrays.toString(s1_array).replaceAll("[,\\[\\] ]", "");
            }
        } else {
            if (input_string.size() != 0) {
                return input_string.get(0);
            } else {
                return "";
            }
        }
    }
    private static List<String> Match_Key_Words(List<String> input_string) {
        got_key_words = false;
        for (int i = 0; i < input_string.size(); i++) {
            boolean english_words = false;
            String[] s1_array = input_string.get(i).split("");
            if (s1_array[0].matches("^[A-Za-z]+$")) {
                s1_array = input_string.get(0).split(" ");
                english_words = true;
            }
            for (int j = 0; j < s1_array.length; j++) {
                for (int k = 0; k < first_key_words.length; k++) {
                    if (Objects.equals(s1_array[j], first_key_words[k])) {
                        if (second_key_words[k].equals(" ")) {
                            got_key_words = true;
                            return input_string.subList(i, input_string.size());
                        } else if (j + 1 < s1_array.length) {
                            if (Objects.equals(s1_array[j + 1], second_key_words[k])) {
                                got_key_words = true;
                                return input_string.subList(i, input_string.size());
                            }
                        } else if ((j == s1_array.length - 1) && (i + 1 < input_string.size())) {
                            String[] s2_array;
                            if (english_words) {
                                s2_array = input_string.get(i + 1).split(" ");
                            } else {
                                s2_array = input_string.get(i + 1).split("");
                            }
                            if (Objects.equals(s2_array[0], second_key_words[k])) {
                                got_key_words = true;
                                s2_array[0] = s1_array[s1_array.length - 1] + s2_array[0];
                                if (english_words) {
                                    input_string.add(i + 2, Arrays.toString(s2_array).replaceAll("[,\\[\\]]", ""));  // Remain white space.
                                } else {
                                    input_string.add(i + 2, Arrays.toString(s2_array).replaceAll("[,\\[\\] ]", ""));
                                }
                                return input_string.subList(i + 2, input_string.size());
                            }
                        }
                    }
                }
            }
        }
        return input_string;
    }
    private void Read_Assets(String file_name, AssetManager mgr) {
        switch (file_name) {
            case file_name_vocab_asr -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_vocab_asr)));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        vocab_asr.add(line);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            case file_name_response -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_response)));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        wake_up_response.add(line);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            case file_name_negMean_asr -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_negMean_asr)));
                    String[] values = reader.readLine().split("\\s+");
                    for (int i = 0; i < values.length; i++) {
                        negMean_asr[i] = Float.parseFloat(values[i]);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            case file_name_negMean_vad -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_negMean_vad)));
                    String[] values = reader.readLine().split("\\s+");
                    for (int i = 0; i < values.length; i++) {
                        negMean_vad[i] = Float.parseFloat(values[i]);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            case file_name_invStd_asr -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_invStd_asr)));
                    String[] values = reader.readLine().split("\\s+");
                    for (int i = 0; i < values.length; i++) {
                        invStd_asr[i] = Float.parseFloat(values[i]);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            case file_name_invStd_vad -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_invStd_vad)));
                    String[] values = reader.readLine().split("\\s+");
                    for (int i = 0; i < values.length; i++) {
                        invStd_vad[i] = Float.parseFloat(values[i]);
                    }
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            case file_name_speakers -> {
                try (FileInputStream inputStream = new FileInputStream(cache_path + file_name_speakers)) {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(Channels.newInputStream(inputStream.getChannel())));
                    String line;
                    int row = 0;
                    while ((line = reader.readLine()) != null && row < score_data_Speaker.length) {
                        String[] stringValues = line.trim().split("\\s+");
                        for (int i = 0; i < model_hidden_size_Res2Net; i++) {
                            score_data_Speaker[row][i] = Float.parseFloat(stringValues[i]);
                        }
                        row++;
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    private void saveToFile(float[][] float2DArray, String filePath) {
        StringBuilder stringBuilder = new StringBuilder();
        for (float[] row : float2DArray) {
            for (float value : row) {
                stringBuilder.append(value);
                stringBuilder.append(" ");
            }
            stringBuilder.append("\n");
        }
        try (FileWriter writer = new FileWriter(filePath)) {
            new File(filePath).getParentFile().mkdirs();
            writer.write(stringBuilder.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private native boolean Load_Models_0(AssetManager assetManager, boolean FP16, boolean USE_GPU, boolean USE_NNAPI, boolean USE_XNNPACK, boolean USE_QNN, boolean USE_DSP_NPU);
    private native boolean Load_Models_1(AssetManager assetManager, boolean FP16, boolean USE_GPU, boolean USE_NNAPI, boolean USE_XNNPACK, boolean USE_QNN, boolean USE_DSP_NPU);
    private native boolean Load_Models_2(AssetManager assetManager, boolean FP16, boolean USE_GPU, boolean USE_NNAPI, boolean USE_XNNPACK, boolean USE_QNN, boolean USE_DSP_NPU);
    private native boolean Pre_Process(float[] neg_mean, float[] inv_std, float[] neg_mean_vad, float[] inv_std_vad);
    private static native int[] Run_VAD_ASR(int record_size_16k, float[] audio, boolean[] arousal_awake, boolean focus_mode, int stop_asr);
    private static native float[] Run_Speaker_Confirm(int mic_id, int model_hidden_size);
}
