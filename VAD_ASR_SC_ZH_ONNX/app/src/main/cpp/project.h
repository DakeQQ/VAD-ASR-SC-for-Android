
#include <jni.h>
#include "onnxruntime_cxx_api.h"
#include "nnapi_provider_factory.h"
#include <android/asset_manager_jni.h>
#include <random>
#include <complex>
#include <iostream>
#include <fstream>

const OrtApi* ort_runtime_A;
OrtSession* session_model_A;
const std::string file_name_A = "Model_VAD.ort";
std::vector<const char*> input_names_A;
std::vector<const char*> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue*> input_tensors_A;
std::vector<OrtValue*> output_tensors_A;
const OrtApi* ort_runtime_B;
OrtSession* session_model_B;
const std::string file_name_B = "Model_Paraformer.ort";
std::vector<const char*> input_names_B;
std::vector<const char*> output_names_B;
std::vector<std::vector<std::int64_t>> input_dims_B;
std::vector<std::vector<std::int64_t>> output_dims_B;
std::vector<ONNXTensorElementDataType> input_types_B;
std::vector<ONNXTensorElementDataType> output_types_B;
std::vector<OrtValue*> input_tensors_B;
std::vector<OrtValue*> output_tensors_B;
const OrtApi* ort_runtime_C;
OrtSession* session_model_C;
const std::string file_name_C = "Model_ERes2Net.ort";
std::vector<const char*> input_names_C;
std::vector<const char*> output_names_C;
std::vector<std::vector<std::int64_t>> input_dims_C;
std::vector<std::vector<std::int64_t>> output_dims_C;
std::vector<ONNXTensorElementDataType> input_types_C;
std::vector<ONNXTensorElementDataType> output_types_C;
std::vector<OrtValue*> input_tensors_C;
std::vector<OrtValue*> output_tensors_C;
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";
const char* qnn_htp_so = "/data/user/0/com.example.myapplication/cache/libQnnHtp.so";  //  If use (std::string + "libQnnHtp.so").c_str() instead, it will open failed.
const char* qnn_cpu_so = "/data/user/0/com.example.myapplication/cache/libQnnCpu.so";  //  If use (std::string + "libQnnCpu.so").c_str() instead, it will open failed.
const int amount_of_vocab = 8359;
const int num_bins_asr = 80;  // number of mel
const int asr_input_shape = 560; // paraformer model accepts the input with shape (1, x, 560). // The same variable in the MainActivity.java, please modify at the same time.
const int vad_input_shape = 400; // VAD model accepts the input with shape (1, x, 400). // The same variable in the MainActivity.java, please modify at the same time.
const int vad_output_shape = 248; // VAD model output shape (1, x, 248)
const int refresh_sliding_window_per_loop = 1;  // The cached feature window is feature_sliding_window_size*(7*80). Specify an Int value to refresh ?*(7*80) features.
const int amount_of_features_compute_per_loop = 7 * refresh_sliding_window_per_loop;   // Get the fbanks features per every input audio, Do not less than 7.
const int fft_points_asr = 512;  // static_cast<int>(pow(2, ceil(log(audio_length) / log(2))))
const int fft_points_half_asr = 257;  // fft_points_asr / 2 + 1
const int feature_sliding_window_size = 15;  //  Must be the same with the exported ASR model.ort.
const int vad_in_cache_size = 9728;  // (4, 1, 128, 19, 1)
const int number_of_history_audio = 2;  // For VAD
const int number_of_frame_state = 5;  // For VAD
const int silent_pdf_ids = 1;  // top_K, The output of VAD is (1, x, 248), takes the top_K/248 into summation.
const int total_elements_in_sliding_window = asr_input_shape * feature_sliding_window_size;  // For 1 batch
const int total_elements_output_vad = vad_output_shape * total_elements_in_sliding_window / vad_input_shape;  // For 1 batch
const int vad_index_offset = total_elements_output_vad - number_of_frame_state * vad_output_shape;
const int BluesteinSequenceLengthThreshold = 46341;
const int amount_of_mic_channel = 1;  // The same variable in the MainActivity.java, please modify at the same time.
const int pre_allocate_num_words = 25;  // The same variable in the MainActivity.java, please modify at the same time.
const int total_elements_in_pre_allocate = amount_of_mic_channel * pre_allocate_num_words;
const int refresh_size = refresh_sliding_window_per_loop * asr_input_shape;
const int asr_buffer_size_0 = total_elements_in_sliding_window * sizeof(float);
const int vad_buffer_size = vad_in_cache_size * sizeof(float);
const int res2net_input_size = 2800;  // 1 * 35 * 80
const int res2net_buffer_size = res2net_input_size * sizeof(float);
int bluesteinConvolution_size_factor_asr;
int noise_count = 1;
const float emphasis_factor = 0.97f; // The smaller value means greater emphasis.
const float sample_rate_asr = 16000.f;
const float speech_2_noise_ratio = 1.f;
const float one_minus_speech_threshold = 0.27f;  // In FunASR FSMN-VAD, it refers to (1 - speech_threshold). The larger value means greater sensitivity, but may induce incorrect ASR results.
const float one_minus_speech_threshold_for_awake = 0.3f;
const float snr_threshold = 5.f;  // Judge if (speech_db - environment_db) >= snr_threshold or not.
const float loop_time = 0.06f;  // unit: second. It used for pre-allocate. Therefore, the value must >= real cost per loop.
const float pi = 3.1415926536f;
const float window_factor_asr = 0.012295862f; // 2 * pi / (frame_length - 1)
const float noise_factor = 1.f;  // noise amplify factor
const float inv_fft_points_asr = 0.001953125f;  //  1 / fft_points_asr
const float inv_16bit_factor = 1.f / (32768.f * 32768.f);  
const float inv_reference_air_pressure_square = 2500000000.f;  //  1 / (0.00002 * 0.00002)
bool remove_dc_offset = false;  // Enable it if you need.
bool add_noise = false;  // Enable it if you need.
std::vector<int32_t> speech_length(1, feature_sliding_window_size);
std::vector<bool> trigger_ASR(amount_of_mic_channel,false);
std::vector<float> neg_mean_asr(asr_input_shape,0.f);
std::vector<float> inv_std_asr(asr_input_shape,0.f);
std::vector<float> neg_mean_vad(vad_input_shape,0.f);
std::vector<float> inv_std_vad(vad_input_shape,0.f);
std::vector<float> Blackman_Harris_factor_asr(fft_points_asr,0.f);
std::vector<float> noise_average_decibel(amount_of_mic_channel,30.f);  // It usually db<=30 in a quiet room.
std::vector<float> white_noise(total_elements_in_sliding_window,0.f);
std::vector<std::vector<float>> features_for_asr(amount_of_mic_channel,std::vector<float> (total_elements_in_sliding_window,0.f));
std::vector<std::vector<float>> features_for_vad(amount_of_mic_channel,std::vector<float> (total_elements_in_sliding_window,0.f));
std::vector<std::vector<float>> features_for_speaker_confirm(amount_of_mic_channel,std::vector<float> (total_elements_in_sliding_window,0.f));
std::vector<std::vector<float>> history_signal(amount_of_mic_channel,std::vector<float> (static_cast<int> (sample_rate_asr * loop_time) * number_of_history_audio, 0.f));
std::vector<std::vector<float>> vad_in_cache(amount_of_mic_channel,std::vector<float> (vad_in_cache_size, 0.f));
std::vector<std::complex<float>> bluesteinsequence_asr;
std::vector<std::complex<float>> bluesteinsequence_b_asr;
std::vector<std::vector<std::complex<float>>> Radix_factor;
std::vector<std::vector<std::complex<float>>> Radix_factor_inv;
std::vector<std::pair<int, std::vector<float>>> fbanks_asr;
