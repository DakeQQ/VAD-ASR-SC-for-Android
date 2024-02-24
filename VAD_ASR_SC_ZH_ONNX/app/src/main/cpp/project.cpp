#include "project.h"

static inline float MelScale(float freq) {
    return 1127.f * std::logf(1.f + freq / 700.f);
}

static inline std::vector<std::pair<int, std::vector<float>>> FBanks(float sp_rate, int fft_points, int fft_points_half, int n_mels) {  // FB-29 Skowronsky-Harris
    float fft_bin_width = sp_rate * inv_fft_points_asr;
    float mel_low_freq = MelScale(20.f);
    float mel_high_freq = MelScale(sp_rate * 0.5f);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / static_cast<float> (n_mels + 1);
    std::vector<std::pair<int, std::vector<float>>> bins;
    bins.resize(n_mels);
    for (int j = 0; j < n_mels; j++) {
        float left_mel = mel_low_freq + static_cast<float>(j) * mel_freq_delta;
        float center_mel = left_mel + mel_freq_delta;
        float right_mel = center_mel + mel_freq_delta;
        std::vector<float> this_bin(fft_points_half, 0.f);
        int first_index = -1;
        int last_index = -1;
        for (int i = 0; i < fft_points_half; ++i) {
            float mel = MelScale(fft_bin_width * static_cast<float> (i));
            if ((mel > left_mel) && (mel < right_mel)) {
                if (mel <= center_mel) {
                    this_bin[i] = (mel - left_mel) / (center_mel - left_mel);
                } else {
                    this_bin[i] = (right_mel - mel) / (right_mel - center_mel);
                }
                if (first_index < 0) {
                    first_index = i;
                }
                last_index = i;
            }
        }
        bins[j].first = first_index;
        int size = last_index + 1 - first_index;
        bins[j].second.resize(size);
        for (int i = 0; i < size; ++i) {
            bins[j].second[i] = this_bin[first_index + i];
        }
    }
    return bins;
}

static inline void PreEmphasis(float coefficient, std::vector<float>* data)  {
    if (coefficient == 0.f) {
        return;
    }
    for (int i = static_cast<int> (data->size()) - 1; i > 0; i--) {
        (*data)[i] -= coefficient * (*data)[i - 1];
    }
    (*data)[0] -= coefficient * (*data)[0];
}

static inline void adding_window(std::vector<std::complex<float>>& data, const std::vector<float>& factor)  {
    for (int i = 0; i < factor.size(); ++i) {
        data[i] *= factor[i];
    }
}

static inline int CelingPowerOf2(int n) {
    --n;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return n + 1;
}

static inline void Radix2Step(std::vector<std::complex<float>>& data, const size_t levelSize, const size_t k, const std::complex<float> factor) {
    size_t step = levelSize << 1;
    for (size_t i = k; i < data.size(); i += step) {
        std::complex<float> ai = data[i];
        std::complex<float> t = factor * data[i + levelSize];
        data[i] = ai + t;
        data[i + levelSize] = ai - t;
    }
}

static inline void Radix2Reorder(std::vector<std::complex<float>>& data) {
    size_t j = 0;
    for (size_t i = 0; i < data.size() - 1; i++) {
        if (i < j) {
            std::complex<float> temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        size_t m = data.size();
        do {
            m >>= 1;
            j ^= m;
        } while ((j & m) == 0);
    }
}

static inline void Radix2Forward(std::vector<std::complex<float>>& data) {
    Radix2Reorder(data);
    int index_levelSize = 0;
    for (size_t levelSize = 1; levelSize < data.size(); levelSize <<= 1) {
        for (size_t k = 0; k < levelSize; k++) {
            Radix2Step(data, levelSize, k, Radix_factor[index_levelSize][k]);
        }
        index_levelSize++;
    }
}

static inline void Radix2Inverse(std::vector<std::complex<float>>& data) {
    Radix2Reorder(data);
    size_t index_levelSize = 0;
    for (size_t levelSize = 1; levelSize < data.size(); levelSize <<= 1) {
        for (size_t k = 0; k < levelSize; k++) {
            Radix2Step(data, levelSize, k,Radix_factor_inv[index_levelSize][k]);
        }
        index_levelSize++;
    }
}

static inline std::vector<std::complex<float>> BluesteinSequence(const int N) {
    float s = pi / static_cast<float> (N);
    std::vector<std::complex<float>> sequence(N);
    if (N > BluesteinSequenceLengthThreshold) {
        for (int k = 0; k < N; k++) {
            float t = (s * static_cast<float> (k)) * static_cast<float> (k);
            sequence[k] = std::polar(1.f, t);
        }
    }
    else {
        for (int k = 0; k < N; k++) {
            float t = s * static_cast<float> (k * k);
            sequence[k] = std::polar(1.f, t);
        }
    }
    return sequence;
}

static inline void BluesteinConvolution(std::vector<std::complex<float>>& data, bool inverse) {
    std::vector<std::complex<float>> bluesteinsequence_a(bluesteinConvolution_size_factor_asr, 0.f);
    for (size_t i = 0; i < data.size(); i++) {
        bluesteinsequence_a[i] = std::conj(bluesteinsequence_asr[i]) * data[i];
    }
    Radix2Forward(bluesteinsequence_a);
    for (size_t i = 0; i < bluesteinConvolution_size_factor_asr; i++) {
        bluesteinsequence_a[i] *= bluesteinsequence_b_asr[i];
    }
    Radix2Inverse(bluesteinsequence_a);
    float nbinv = 1.f / static_cast<float>(bluesteinConvolution_size_factor_asr);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = nbinv * std::conj(bluesteinsequence_asr[i]) * bluesteinsequence_a[i];
    }
}

static inline void BluesteinForward(std::vector<std::complex<float>>& data) {
    BluesteinConvolution(data,false);
}

static inline void BluesteinInverse(std::vector<std::complex<float>>& data) {
    for (auto & i : data) {
        i = std::complex(i.imag(),i.real());
    }
    BluesteinConvolution(data,true);
    for (auto & i : data) {
        i = std::complex(i.imag(), i.real());
    }
}

static inline void FFT(std::vector<std::complex<float>>& data, bool use_Radix2Forward) {
    if (use_Radix2Forward) {
        Radix2Forward(data);
    } else {
        BluesteinForward(data); // Bluestein method is for non 2^n FFT points
    }
}

static inline void IFFT(std::vector<std::complex<float>>& data, bool use_Radix2Inverse) {
    if (use_Radix2Inverse) {
        Radix2Inverse(data);
        for (auto & i : data) {
            i *= inv_fft_points_asr;
        }
    } else {
        BluesteinInverse(data);  // Bluestein method is for non 2^n FFT points
    }
}

static inline int Compute_Features(const int n_mels, const int fft_points, const int target_amount_features, const float inv_factor, bool noise, bool dc_offset, std::vector<float> &wave_points, const std::vector<std::pair<int, std::vector<float>>> &fbank, std::vector<float> &save_fearures, bool use_Radix2Inverse) {
    if (noise) {
        for (float & j : wave_points) {
            std::mt19937 gen{std::random_device{}()};
            std::normal_distribution<float> Gauss_matrix{0.f, 1.f};
            j += noise_factor * Gauss_matrix(gen);
        }
    }
    if (dc_offset) {
        float mean = 0.f;
        for (float j : wave_points) {
            mean += j;
        }
        mean /= static_cast<float> (wave_points.size());
        for (float & j : wave_points) {
            j -= mean;
        }
    }
    PreEmphasis(emphasis_factor, &wave_points);
    int target_sliding_length = (static_cast<int> (wave_points.size()) - fft_points) / (target_amount_features - 1);
    std::vector<std::complex<float>> fft_data(fft_points, 0.f);
    int i_index = 0;
    int j_index = 0;
    for (int i = 0; i < target_amount_features; i++) {
        std::copy(wave_points.begin() + i_index,wave_points.begin() + i_index + fft_points,fft_data.begin());
        adding_window(fft_data,Blackman_Harris_factor_asr);
        FFT(fft_data,use_Radix2Inverse);
        for (int j = 0; j < n_mels; ++j) {
            float mel_energy = 0.f;
            for (int k = 0; k < fbank[j].second.size(); ++k) {
                mel_energy += fbank[j].second[k] * std::norm(fft_data[fbank[j].first + k]);
            }
            if (mel_energy < std::numeric_limits<float>::epsilon()) {
                mel_energy = std::numeric_limits<float>::epsilon();
            }
            save_fearures[j_index + j] = std::logf(mel_energy * inv_factor);
        }
        i_index += target_sliding_length;
        j_index += n_mels;
    }
    return target_sliding_length;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_10(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_fp16,
                                                            jboolean use_gpu,
                                                            jboolean use_nnapi,
                                                            jboolean use_xnnpack,
                                                            jboolean use_qnn,
                                                            jboolean use_dsp_npu) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_A;
    OrtSessionOptions *session_options_A;
    {
        std::vector<char> fileBuffer;
        size_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_A.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_A, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_A->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_A);
        ort_runtime_A->CreateSessionOptions(&session_options_A);
        ort_runtime_A->DisableProfiling(session_options_A);
        ort_runtime_A->EnableCpuMemArena(session_options_A);
        ort_runtime_A->EnableMemPattern(session_options_A);
        ort_runtime_A->SetSessionExecutionMode(session_options_A, ORT_SEQUENTIAL);
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 2);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,  // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "5;7");  // We set two small-core here, although one small-core is fast enough for FSMN-VAD model.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 3); // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.enable_gelu_approximation",
                                             "0");  // Set 0 is better for this model
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.set_denormal_as_zero",
                                             "0");  // // Use 0 instead of NaN or Inf.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_qnn) {  // It needs the permission of HTP hardware, and then follow the onnx document to generate the specific format to run on HTP.
            if (use_dsp_npu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_htp_so);
                option_keys.push_back("htp_performance_mode");
                option_values.push_back("burst");
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");
                option_keys.push_back("soc_model");
                option_values.push_back("0");  // 0 for unknown
                option_keys.push_back("htp_arch");
                option_values.push_back("73");  // 0 for unknown
                option_keys.push_back("device_id");
                option_values.push_back("0");  // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("8");  // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "ep.context_enable", "1");
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "ep.context_embed_mode", "1");
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "ep.context_file_path", storage_path.c_str());  // Default to original_file_name_ctx.onnx if not specified
                ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                     "session.use_ort_model_bytes_directly",
                                                     "0");  // Cancel this option.
            } else {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_cpu_so);
            }
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "QNN", option_keys.data(), option_values.data(), option_keys.size());
        } else if (use_nnapi) {  // It needs to add the app into /vendor/etc/nnapi_extensions_app_allowlist
            uint32_t npflags = 0;
            if (use_gpu | use_dsp_npu) {
                npflags |= NNAPI_FLAG_CPU_DISABLED;
            } else {
                npflags |= NNAPI_FLAG_CPU_ONLY;
            }
            if (use_fp16) {
                npflags |= NNAPI_FLAG_USE_FP16;
            }
            OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options_A, npflags);
        } else if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_A->CreateSessionFromArray(ort_env_A, fileBuffer.data(), fileSize,
                                                       session_options_A, &session_model_A);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_A->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_A->SessionGetInputCount(session_model_A, &amount_of_input);
    input_names_A.resize(amount_of_input);
    input_dims_A.resize(amount_of_input);
    input_types_A.resize(amount_of_input);
    input_tensors_A.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetInputName(session_model_A, i, allocator, &name);
        input_names_A[i] = name;
        ort_runtime_A->SessionGetInputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        input_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, input_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_A->SessionGetOutputCount(session_model_A, &amount_of_output);
    output_names_A.resize(amount_of_output);
    output_dims_A.resize(amount_of_output);
    output_types_A.resize(amount_of_output);
    output_tensors_A.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetOutputName(session_model_A, i, allocator, &name);
        output_names_A[i] = name;
        ort_runtime_A->SessionGetOutputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        output_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, output_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_11(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_fp16,
                                                            jboolean use_gpu,
                                                            jboolean use_nnapi,
                                                            jboolean use_xnnpack,
                                                            jboolean use_qnn,
                                                            jboolean use_dsp_npu) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_B;
    OrtSessionOptions *session_options_B;
    {
        std::vector<char> fileBuffer;
        size_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_B.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_B, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_B = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_B->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_B);
        ort_runtime_B->CreateSessionOptions(&session_options_B);
        ort_runtime_B->DisableProfiling(session_options_B);
        ort_runtime_B->EnableCpuMemArena(session_options_B);
        ort_runtime_B->EnableMemPattern(session_options_B);
        ort_runtime_B->SetSessionExecutionMode(session_options_B, ORT_SEQUENTIAL);
        ort_runtime_B->SetInterOpNumThreads(session_options_B, 2);
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base", "2");
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.intra_op_thread_affinities",
                                             "6;8");  // We set two small-core here, although one small-core is fast enough for paraformer model.
        ort_runtime_B->SetIntraOpNumThreads(session_options_B, 3); // dynamic_block_base + 1
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_B->SetSessionGraphOptimizationLevel(session_options_B, ORT_ENABLE_ALL);
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "optimization.enable_gelu_approximation",
                                             "0");  // Set 0 is better for this model
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.set_denormal_as_zero",
                                             "0");  // // Use 0 instead of NaN or Inf.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_qnn) {  // It needs the permission of HTP hardware, and then follow the onnx document to generate the specific format to run on HTP.
            if (use_dsp_npu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_htp_so);
                option_keys.push_back("htp_performance_mode");
                option_values.push_back("burst");
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");
                option_keys.push_back("soc_model");
                option_values.push_back("0");  // 0 for unknown
                option_keys.push_back("htp_arch");
                option_values.push_back("73");  // 0 for unknown
                option_keys.push_back("device_id");
                option_values.push_back("0");  // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("8");  // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "ep.context_enable", "1");
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "ep.context_embed_mode", "1");
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "ep.context_file_path", storage_path.c_str());  // Default to original_file_name_ctx.onnx if not specified
                ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                                     "session.use_ort_model_bytes_directly",
                                                     "0");  // Cancel this option.
            } else {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_cpu_so);
            }
            ort_runtime_B->SessionOptionsAppendExecutionProvider(session_options_B, "QNN", option_keys.data(), option_values.data(), option_keys.size());
        } else if (use_nnapi) {  // It needs to add the app into /vendor/etc/nnapi_extensions_app_allowlist
            uint32_t npflags = 0;
            if (use_gpu | use_dsp_npu) {
                npflags |= NNAPI_FLAG_CPU_DISABLED;
            } else {
                npflags |= NNAPI_FLAG_CPU_ONLY;
            }
            if (use_fp16) {
                npflags |= NNAPI_FLAG_USE_FP16;
            }
            OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options_B, npflags);
        } else if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_B->SessionOptionsAppendExecutionProvider(session_options_B, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_B->CreateSessionFromArray(ort_env_B, fileBuffer.data(), fileSize,
                                                       session_options_B, &session_model_B);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_B->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_B->SessionGetInputCount(session_model_B, &amount_of_input);
    input_names_B.resize(amount_of_input);
    input_dims_B.resize(amount_of_input);
    input_types_B.resize(amount_of_input);
    input_tensors_B.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetInputName(session_model_B, i, allocator, &name);
        input_names_B[i] = name;
        ort_runtime_B->SessionGetInputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        input_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, input_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_B->SessionGetOutputCount(session_model_B, &amount_of_output);
    output_names_B.resize(amount_of_output);
    output_dims_B.resize(amount_of_output);
    output_types_B.resize(amount_of_output);
    output_tensors_B.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetOutputName(session_model_B, i, allocator, &name);
        output_names_B[i] = name;
        ort_runtime_B->SessionGetOutputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        output_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, output_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_12(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_fp16,
                                                            jboolean use_gpu,
                                                            jboolean use_nnapi,
                                                            jboolean use_xnnpack,
                                                            jboolean use_qnn,
                                                            jboolean use_dsp_npu) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_C;
    OrtSessionOptions *session_options_C;
    {
        std::vector<char> fileBuffer;
        size_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_C.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_C, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_C = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_C->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_C);
        ort_runtime_C->CreateSessionOptions(&session_options_C);
        ort_runtime_C->DisableProfiling(session_options_C);
        ort_runtime_C->EnableCpuMemArena(session_options_C);
        ort_runtime_C->EnableMemPattern(session_options_C);
        ort_runtime_C->SetSessionExecutionMode(session_options_C, ORT_SEQUENTIAL);
        ort_runtime_C->SetInterOpNumThreads(session_options_C, 2);
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.dynamic_block_base", "2");
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.intra_op_thread_affinities",
                                             "5;7");  // We set two small-core here, although one small-core is fast enough for Res2Net model.
        ort_runtime_C->SetIntraOpNumThreads(session_options_C, 3); // dynamic_block_base + 1
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_C->SetSessionGraphOptimizationLevel(session_options_C, ORT_ENABLE_ALL);
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "optimization.enable_gelu_approximation",
                                             "0");  // Set 0 is better for this model
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.set_denormal_as_zero",
                                             "0");  // // Use 0 instead of NaN or Inf.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_qnn) {  // It needs the permission of HTP hardware, and then follow the onnx document to generate the specific format to run on HTP.
            if (use_dsp_npu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_htp_so);
                option_keys.push_back("htp_performance_mode");
                option_values.push_back("burst");
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");
                option_keys.push_back("soc_model");
                option_values.push_back("0");  // 0 for unknown
                option_keys.push_back("htp_arch");
                option_values.push_back("73");  // 0 for unknown
                option_keys.push_back("device_id");
                option_values.push_back("0");  // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("8");  // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
                ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                                     "ep.context_enable", "1");
                ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                                     "ep.context_embed_mode", "1");
                ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                                     "ep.context_file_path", storage_path.c_str());  // Default to original_file_name_ctx.onnx if not specified
                ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                                     "session.use_ort_model_bytes_directly",
                                                     "0");  // Cancel this option.
            } else {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_cpu_so);
            }
            ort_runtime_C->SessionOptionsAppendExecutionProvider(session_options_C, "QNN", option_keys.data(), option_values.data(), option_keys.size());
        } else if (use_nnapi) {  // It needs to add the app into /vendor/etc/nnapi_extensions_app_allowlist
            uint32_t npflags = 0;
            if (use_gpu | use_dsp_npu) {
                npflags |= NNAPI_FLAG_CPU_DISABLED;
            } else {
                npflags |= NNAPI_FLAG_CPU_ONLY;
            }
            if (use_fp16) {
                npflags |= NNAPI_FLAG_USE_FP16;
            }
            OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options_C, npflags);
        } else if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_C->SessionOptionsAppendExecutionProvider(session_options_C, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_C->CreateSessionFromArray(ort_env_C, fileBuffer.data(), fileSize,
                                                       session_options_C, &session_model_C);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_C->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_C->SessionGetInputCount(session_model_C, &amount_of_input);
    input_names_C.resize(amount_of_input);
    input_dims_C.resize(amount_of_input);
    input_types_C.resize(amount_of_input);
    input_tensors_C.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_C->SessionGetInputName(session_model_C, i, allocator, &name);
        input_names_C[i] = name;
        ort_runtime_C->SessionGetInputTypeInfo(session_model_C, i, &typeinfo);
        ort_runtime_C->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_C->GetTensorElementType(tensor_info, &type);
        input_types_C[i] = type;
        ort_runtime_C->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_C[i].resize(dimensions);
        ort_runtime_C->GetDimensions(tensor_info, input_dims_C[i].data(), dimensions);
        ort_runtime_C->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_C->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_C->SessionGetOutputCount(session_model_C, &amount_of_output);
    output_names_C.resize(amount_of_output);
    output_dims_C.resize(amount_of_output);
    output_types_C.resize(amount_of_output);
    output_tensors_C.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_C->SessionGetOutputName(session_model_C, i, allocator, &name);
        output_names_C[i] = name;
        ort_runtime_C->SessionGetOutputTypeInfo(session_model_C, i, &typeinfo);
        ort_runtime_C->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_C->GetTensorElementType(tensor_info, &type);
        output_types_C[i] = type;
        ort_runtime_C->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_C[i].resize(dimensions);
        ort_runtime_C->GetDimensions(tensor_info, output_dims_C[i].data(), dimensions);
        ort_runtime_C->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) ort_runtime_C->ReleaseTypeInfo(typeinfo);
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv *env, jobject clazz,
                                                             jfloatArray jneg_mean_asr, jfloatArray jinv_std_asr,
                                                             jfloatArray jneg_mean_vad,
                                                             jfloatArray jinv_std_vad) {
    jfloat *negMean = env->GetFloatArrayElements(jneg_mean_asr, nullptr);
    jfloat *invStd = env->GetFloatArrayElements(jinv_std_asr, nullptr);
    jfloat *negMean_vad = env->GetFloatArrayElements(jneg_mean_vad, nullptr);
    jfloat *invStd_vad = env->GetFloatArrayElements(jinv_std_vad, nullptr);
    std::move(negMean, negMean + asr_input_shape, neg_mean_asr.begin());
    std::move(invStd, invStd + asr_input_shape, inv_std_asr.begin());
    std::move(negMean_vad, negMean_vad + vad_input_shape, neg_mean_vad.begin());
    std::move(invStd_vad, invStd_vad + vad_input_shape, inv_std_vad.begin());
    env->ReleaseFloatArrayElements(jneg_mean_asr, negMean, 0);
    env->ReleaseFloatArrayElements(jinv_std_asr, invStd, 0);
    env->ReleaseFloatArrayElements(jneg_mean_vad, negMean_vad, 0);
    env->ReleaseFloatArrayElements(jinv_std_vad, invStd_vad, 0);
    for (int i = 0; i < fft_points_asr; i++) {
        Blackman_Harris_factor_asr[i] =
                0.35875f - 0.48829f * std::cosf(static_cast<float> (i) * window_factor_asr) +
                0.14128f * std::cosf(static_cast<float> (i) * 2.f * window_factor_asr) -
                0.01168f * std::cosf(static_cast<float> (i) * 3.f * window_factor_asr);
    }
    fbanks_asr = FBanks(sample_rate_asr, fft_points_asr, fft_points_half_asr, num_bins_asr);
    bluesteinConvolution_size_factor_asr = CelingPowerOf2((fft_points_asr << 1) - 1);
    Radix_factor.resize(12, std::vector<std::complex<float>>(bluesteinConvolution_size_factor_asr,0.f));  //  for 4096, 2^12
    int index = 0;
    for (int i = 1; i < bluesteinConvolution_size_factor_asr; i <<= 1) {
        for (int j = 0; j < bluesteinConvolution_size_factor_asr; j++) {
            float temp = static_cast<float> (-j) * pi / static_cast<float> (i);
            Radix_factor[index][j] = std::polar(1.f, temp);
        }
        index++;
    }
    for (int i = 0; i < asr_input_shape; i++) {
        white_noise[i] = neg_mean_asr[i] * inv_std_asr[i];
    }
    for (int i = asr_input_shape; i < total_elements_in_sliding_window; i+=asr_input_shape) {
        std::copy(white_noise.begin(), white_noise.begin() + asr_input_shape, white_noise.begin() + i);
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1VAD_1ASR(JNIEnv *env, jclass clazz,
                                                              jint audio_length,
                                                              jfloatArray audio,
                                                              jbooleanArray jarousal_awake,
                                                              jboolean jfocus_mode, jint jstop_asr) {
    jboolean* awake_channel = env->GetBooleanArrayElements(jarousal_awake, nullptr);
    std::vector<std::vector<float>> resample_signal(amount_of_mic_channel,std::vector<float> (audio_length, 0.f));
    {
        jfloat* waves = env->GetFloatArrayElements(audio, nullptr);
        int mic_shift = 0;
        for (int i = 0; i < amount_of_mic_channel; i++) {
            std::move(waves + mic_shift, waves + mic_shift + audio_length,
                      resample_signal[i].begin());
            mic_shift += audio_length;
        }
        env->ReleaseFloatArrayElements(audio, waves, 0);
    }
    {
        for (int i = 0; i < amount_of_mic_channel; i++) {
            std::vector<std::vector<float>> save_features(amount_of_mic_channel,
                                                          std::vector<float>(refresh_size,0.f));
            Compute_Features(num_bins_asr,fft_points_asr,amount_of_features_compute_per_loop,
                             inv_fft_points_asr,add_noise,remove_dc_offset,resample_signal[i],
                             fbanks_asr, save_features[i],true);
            std::move(features_for_speaker_confirm[i].begin() + refresh_size,features_for_speaker_confirm[i].end(),
                      features_for_speaker_confirm[i].begin());
            std::copy(save_features[i].begin(),save_features[i].end(),
                      features_for_speaker_confirm[i].end() - refresh_size);
            if (i != jstop_asr) {
                std::move(features_for_asr[i].begin() + refresh_size,features_for_asr[i].end(),
                          features_for_asr[i].begin());
                std::move(features_for_vad[i].begin() + refresh_size,features_for_vad[i].end(),
                          features_for_vad[i].begin());
                std::copy(save_features[i].begin(),save_features[i].end(),
                          features_for_vad[i].end() - refresh_size);
                std::move(save_features[i].begin(), save_features[i].end(),
                          features_for_asr[i].end() - refresh_size);
                int index_asr = 0;
                int index_vad = 0;
                for (int j = total_elements_in_sliding_window - refresh_size;
                     j < total_elements_in_sliding_window; j++) {
                    features_for_asr[i][j] =
                            (features_for_asr[i][j] + neg_mean_asr[index_asr]) * inv_std_asr[index_asr];
                    features_for_vad[i][j] =
                            (features_for_vad[i][j] + neg_mean_vad[index_vad]) *
                            inv_std_vad[index_vad];
                    index_asr++;
                    index_vad++;
                    if (index_asr >= asr_input_shape) {
                        index_asr = 0;
                    }
                    if (index_vad >= vad_input_shape) {
                        index_vad = 0;
                    }
                }
            } else {
                std::copy(white_noise.begin(), white_noise.end(), features_for_asr[i].begin());
                std::fill(features_for_vad[i].begin(), features_for_vad[i].end(), 0.f);
            }
        }
    }
    for (int i = 0; i < amount_of_mic_channel; i++) {
        if (i != jstop_asr) {
            std::move(history_signal[i].begin() + audio_length, history_signal[i].begin() + number_of_history_audio * audio_length, history_signal[i].begin());
            std::move(resample_signal[i].begin(), resample_signal[i].end(), history_signal[i].begin() + (number_of_history_audio - 1) * audio_length);
            std::destroy(resample_signal[i].begin(), resample_signal[i].end());
        }
        else {
            std::fill(history_signal[i].begin(), history_signal[i].end(), 0.f);
        }
    }
    bool monitoring_all = true;
    if (jfocus_mode) {
        for (int k = 0; k < amount_of_mic_channel; k++) {
            if (awake_channel[k]) {
                monitoring_all = false;
                break;
            }
        }
    }
    std::vector<int> frame_state(amount_of_mic_channel,0);  // 0 for silent, 1 for speaking
    {
        int hop_size = (number_of_history_audio - 1) * audio_length / (number_of_frame_state - 1);
        float speech_threshold;
        float inv_reference_factor = inv_reference_air_pressure_square / static_cast<float> (audio_length);
        for (int k = 0; k < amount_of_mic_channel; k++) {
            if (k != jstop_asr) {
                if (monitoring_all | awake_channel[k]) {
                    if (awake_channel[k]) {
                        speech_threshold = one_minus_speech_threshold;
                    } else {
                        speech_threshold = one_minus_speech_threshold_for_awake;
                    }
                    void* output_tensors_buffer_0;
                    void* output_tensors_buffer_1;
                    OrtMemoryInfo *memory_info;
                    ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
                    ort_runtime_A->CreateTensorWithDataAsOrtValue(
                            memory_info, reinterpret_cast<void*>(features_for_vad[k].data()), total_elements_in_sliding_window * sizeof(float),
                            input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0], &input_tensors_A[0]);
                    ort_runtime_A->CreateTensorWithDataAsOrtValue(
                            memory_info, reinterpret_cast<void*>(vad_in_cache[k].data()), vad_buffer_size,
                            input_dims_A[1].data(), input_dims_A[1].size(), input_types_A[1], &input_tensors_A[1]);
                    ort_runtime_A->ReleaseMemoryInfo(memory_info);
                    ort_runtime_A->Run(session_model_A, nullptr, input_names_A.data(), (const OrtValue* const*)input_tensors_A.data(),
                                       input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                                       output_tensors_A.data());
                    ort_runtime_A->GetTensorMutableData(output_tensors_A[0], &output_tensors_buffer_0);
                    ort_runtime_A->GetTensorMutableData(output_tensors_A[1], &output_tensors_buffer_1);
                    std::move(reinterpret_cast<float*> (output_tensors_buffer_1), reinterpret_cast<float*> (output_tensors_buffer_1) + vad_in_cache_size, vad_in_cache[k].begin());
                    size_t index_i = 0;
                    size_t index_j = vad_index_offset;
                    for (int i = 0; i < number_of_frame_state; i++) {
                        float sum = 0.f;
                        for (size_t j = index_i; j < index_i + audio_length; j++) {
                            sum += history_signal[k][j] * history_signal[k][j];
                        }
                        float cur_decibel = 10.f * std::log10f(sum * inv_reference_factor * inv_16bit_factor + 0.00002f);  // avoid log(0)
                        float sum_score = 0.f;
                        for (size_t j = index_j; j < index_j + silent_pdf_ids; j++) {
                            sum_score += reinterpret_cast<float*> (output_tensors_buffer_0)[j];
                        }
                        if (speech_2_noise_ratio == 1.f) {
                            sum_score += sum_score;
                        } else if (speech_2_noise_ratio > 1.f) {
                            sum_score = std::powf(sum_score, speech_2_noise_ratio) + sum_score;
                        } else {
                            sum_score += 1.f;
                        }
                        if (sum_score <= speech_threshold) { // be simplified. The original expression: exp(log(1 - sum_score)) >= exp(speech_2_noise_ratio * log(sum_score)) + speech_threshold
                            if (noise_count * (cur_decibel - snr_threshold) >= noise_average_decibel[k]) {
                                frame_state[k] += 1;  // +1 means current frame is judged to activation state.
                                if (jarousal_awake) {
                                    if (frame_state[k] > 2) {  // '2' it is a editable value. Modify if you need to adjust the VAD de-activate sensitivity.
                                        break;
                                    }
                                } else {
                                    if (frame_state[k] > 0) {  // '0' it is a editable value. Modify if you need to adjust the VAD activate sensitivity.
                                        break;
                                    }
                                }
                            } else if (trigger_ASR[k]) {
                                break;
                            }
                        } else {
                            noise_count++;
                            noise_average_decibel[k] += cur_decibel;
                            if (noise_count > 999) {  // Take only the recent average noise dB into account.
                                noise_average_decibel[k] /= noise_count;
                                noise_average_decibel[k] *= 500;
                                noise_count = 500;
                            }
                        }
                        index_i += hop_size;
                        index_j += vad_output_shape;
                    }
                }
            }
        }
    }
    for (int i = 0; i < amount_of_mic_channel; i++) {
        if (i != jstop_asr) {
            if (monitoring_all | awake_channel[i]) {
                if (trigger_ASR[i]) {
                    if (frame_state[i] < number_of_frame_state) {
                        trigger_ASR[i] = false;
                    }
                } else {
                    if (awake_channel[i]) {
                        if (frame_state[i] > 2) {   // '2' it is a editable value. Modify if you need to adjust the VAD de-activate sensitivity.
                            trigger_ASR[i] = true;
                        }
                    } else {
                        if (frame_state[i] > 0) {  // '0' it is a editable value. Modify if you need to adjust the VAD activate sensitivity.
                            trigger_ASR[i] = true;
                        }
                    }
                }
            } else {
                trigger_ASR[i] = false;
            }
        } else {
            trigger_ASR[i] = false;
        }
    }
    std::vector<int> max_position(total_elements_in_pre_allocate, -1);
    {
        int index_k = 0;
        for (int k = 0; k < amount_of_mic_channel; k++) {
            if (trigger_ASR[k]) {
                void *output_tensors_buffer_0;
                void *output_tensors_buffer_1;
                OrtMemoryInfo *memory_info;
                ort_runtime_B->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                                   &memory_info);
                ort_runtime_B->CreateTensorWithDataAsOrtValue(
                        memory_info, reinterpret_cast<void *>(features_for_asr[k].data()),
                        asr_buffer_size_0,
                        input_dims_B[0].data(), input_dims_B[0].size(), input_types_B[0],
                        &input_tensors_B[0]);
                ort_runtime_B->CreateTensorWithDataAsOrtValue(
                        memory_info, reinterpret_cast<void *>(speech_length.data()), sizeof(int32_t),
                        input_dims_B[1].data(), input_dims_B[1].size(), input_types_B[1],
                        &input_tensors_B[1]);
                ort_runtime_B->ReleaseMemoryInfo(memory_info);
                ort_runtime_B->Run(session_model_B, nullptr, input_names_B.data(),
                                   (const OrtValue *const *) input_tensors_B.data(),
                                   input_tensors_B.size(), output_names_B.data(),
                                   output_names_B.size(),
                                   output_tensors_B.data());
                ort_runtime_B->GetTensorMutableData(output_tensors_B[0], &output_tensors_buffer_0);
                ort_runtime_B->GetTensorMutableData(output_tensors_B[1], &output_tensors_buffer_1);
                auto *logit = reinterpret_cast<float*> (output_tensors_buffer_0);
                int num_words = reinterpret_cast<int32_t*> (output_tensors_buffer_1)[0];
                if (num_words > pre_allocate_num_words) {
                    num_words = pre_allocate_num_words;
                }
                if (num_words > 0) {
                    int index_i = 0;
                    for (int i = 0; i < num_words; i++) {  // We have tried using 'omp parallel for', but it did not improve performance because 'num_words' is not large enough.
                        float max_values = -999999999.f;
                        for (int j = 0; j < amount_of_vocab; j++) {
                            float value = logit[index_i + j];
                            if (value > max_values) {
                                max_values = value;
                                max_position[i + index_k] = j;
                            }
                        }
                        index_i += amount_of_vocab;
                    }
                }
            }
            index_k += pre_allocate_num_words;
        }
    }
    jintArray final_results = env->NewIntArray(total_elements_in_pre_allocate);
    env->SetIntArrayRegion(final_results, 0, total_elements_in_pre_allocate,max_position.data());
    return final_results;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MainActivity_Run_1Speaker_1Confirm(JNIEnv *env, jclass clazz,
                                                                  jint mic_id,
                                                                  jint model_hidden_size) {
    OrtMemoryInfo *memory_info;
    ort_runtime_C->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_C->CreateTensorWithDataAsOrtValue(
            memory_info, reinterpret_cast<void*>(&(*(features_for_speaker_confirm[mic_id].end() - res2net_input_size))), res2net_buffer_size,
            input_dims_C[0].data(), input_dims_C[0].size(), input_types_C[0], &input_tensors_C[0]);
    ort_runtime_C->Run(session_model_C, nullptr, input_names_C.data(), (const OrtValue* const*) input_tensors_C.data(),
                       input_tensors_C.size(), output_names_C.data(), output_names_C.size(),
                       output_tensors_C.data());
    void* output_tensors_buffer_0;
    ort_runtime_C->GetTensorMutableData(output_tensors_C[0], &output_tensors_buffer_0);
    jfloatArray final_results = env->NewFloatArray(model_hidden_size);
    env->SetFloatArrayRegion(final_results,0,model_hidden_size,reinterpret_cast<float*> (output_tensors_buffer_0));
    return final_results;
}
