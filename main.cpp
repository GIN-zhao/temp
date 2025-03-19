#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <ctime>
namespace fs = std::filesystem;

// Ԥ������ȡͼƬ����һ�������롢ת�����ݸ�ʽ��HWC->CHW��
// ���������image_path - ͼƬ·����n_channels - ͨ������noise_level - ������׼���� uint8 ��Χ 0~255����use_clip �Ƿ������ͼ�����ü���
// ����ֵ������õ����� tensor ���ݣ�ͬʱ���� input_shape����ʽΪ {1, C, H, W}��
std::vector<float> preprocess_image_vector(const std::string& image_path, int n_channels, int noise_level, bool use_clip, std::vector<int64_t>& input_shape) {
    cv::Mat img;
    if (n_channels == 1) {
        img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            throw std::runtime_error("Fail to read image " + image_path);
        }
        // �̶���С
        cv::resize(img, img, cv::Size(512, 512));
    }
    else {
        img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Fail to read image " + image_path);
        }
        // ת�� BGR �� RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    // ת��Ϊ float ����һ���� [0,1]
    img.convertTo(img, CV_32FC(n_channels), 1.0 / 255.0);

    // ��Ӹ�˹��������׼�� = noise_level/255.0
    float noise_std = noise_level / 255.0f;
    cv::Mat noise(img.size(), img.type());
    cv::randn(noise, 0, noise_std);
    cv::Mat img_noisy = img + noise;
    if (use_clip) {
        cv::min(img_noisy, 1.0, img_noisy);
        cv::max(img_noisy, 0.0, img_noisy);
    }

    // HWC ת CHW���ȷ���ͨ�����ٰ�˳����� vector ��
    int height = img_noisy.rows;
    int width = img_noisy.cols;
    int channels = img_noisy.channels();
    input_shape = { 1, channels, height, width };

    std::vector<cv::Mat> split_channels;
    cv::split(img_noisy, split_channels);
    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(channels * height * width);
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                input_tensor_values.push_back(split_channels[c].at<float>(i, j));
            }
        }
    }
    return input_tensor_values;
}

// ���������������tensor �� float ���飩ת��Ϊ cv::Mat ͼ�񣬲�ת��Ϊ uint8 ��ʽ����
// output_shape ��ʽΪ {1, channels, height, width}
cv::Mat postprocess_image(const std::vector<float>& output_tensor, const std::vector<int64_t>& output_shape, int n_channels) {
    int height = static_cast<int>(output_shape[2]);
    int width = static_cast<int>(output_shape[3]);
    int channels = static_cast<int>(output_shape[1]);

    // ����ͨ�����ݸ��Ƶ� cv::Mat ��
    std::vector<cv::Mat> channelMats;
    size_t offset = 0;
    for (int c = 0; c < channels; c++) {
        cv::Mat channel(height, width, CV_32F);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                channel.at<float>(i, j) = output_tensor[offset++];
            }
        }
        channelMats.push_back(channel);
    }
    cv::Mat merged;
    cv::merge(channelMats, merged);
    // Clip �� [0,1] ��ת��Ϊ [0,255]
    cv::threshold(merged, merged, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(merged, merged, 0.0, 0.0, cv::THRESH_TOZERO);
    merged = merged * 255.0;
    merged.convertTo(merged, CV_8U);
    if (n_channels != 1) {
        // �� RGB ת��Ϊ BGR��ȷ�� cv::imwrite ��������
        cv::cvtColor(merged, merged, cv::COLOR_RGB2BGR);
    }
    return merged;
}

// ��ȡ��ǰʱ����ַ�������ʽΪ��YYYYMMDD_HHMMSS
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm time_info;
    localtime_s(&time_info, &now_time_t); // Correct usage of localtime_s
    std::stringstream ss;
    ss << std::put_time(&time_info, "%Y%m%d_%H%M%S");
    return ss.str();
}

// �������ڴ���ָ���ļ����е�ͼƬ
void process_folder(const std::string& folder_path, const std::string& output_dir, int noise_level,
    const std::string& model_name, bool use_clip, int n_channels) {
    // �������Ŀ¼
    fs::create_directories(output_dir);

    // ��ȡ���ļ���������ͼƬ
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                image_paths.push_back(entry.path().string());
            }
        }
    }

    if (image_paths.empty()) {
        std::cout << "Ŀ¼��δ����ͼƬ������·��: " << folder_path << std::endl;
        return;
    }

    // ��ʼ�� ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

    // ��������ÿһ��ͼƬ
    for (const auto& img_path : image_paths) {
        std::cout << "���ڴ���ͼƬ: " << img_path << std::endl;

        // ʹ�� do-while ѭ����ȷ���ڳ���ʱ����
        bool success = false;
        int retry_count = 0;
        const int max_retries = 10; // ������Դ���

        do {
            try {
                if (retry_count > 0) {
                    std::cout << "�� " << retry_count << " �����Դ���ͼƬ: " << img_path << std::endl;
                }

                Ort::Session session{ env, L"ffdnet_gray_clip.onnx", Ort::SessionOptions{ nullptr } };

                // ��ȡ��������
                Ort::AllocatorWithDefaultOptions allocator;
                Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
                Ort::AllocatedStringPtr sigma_name_Ptr = session.GetInputNameAllocated(1, allocator);

                // ���� sigma����״Ϊ (1, 1, 1, 1)
                std::vector<int64_t> sigma_shape = { 1, 1, 1, 1 };
                std::vector<float> sigma_val(1, noise_level / 255.0f);

                // ��ȡ�����������
                size_t num_outputs = session.GetOutputCount();
                std::vector<const char*> output_names;
                for (size_t i = 0; i < num_outputs; i++) {
                    output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
                }

                auto start = std::chrono::high_resolution_clock::now();

                // Ԥ������ȡ����һ�������룬��ת��Ϊ CHW ��ʽ
                std::vector<int64_t> input_shape;
                std::vector<float> input_tensor = preprocess_image_vector(img_path, n_channels, 25, use_clip, input_shape);

                // �������� tensor ����
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());
                Ort::Value sigma_tensor = Ort::Value::CreateTensor<float>(memory_info, sigma_val.data(), sigma_val.size(),
                    sigma_shape.data(), sigma_shape.size());

                // ִ������
                const char* input_names_arr[] = { input_name_Ptr.get(), sigma_name_Ptr.get() };
                auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names_arr,
                    std::array<Ort::Value, 2>{std::move(input_tensor_ort), std::move(sigma_tensor)}.data(),
                    2, output_names.data(), output_names.size());

                // ��ȡ��� tensor ���ݼ���״
                float* output_data = output_tensors.front().GetTensorMutableData<float>();
                auto output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
                std::vector<int64_t> output_shape = output_info.GetShape();
                size_t total_output_elements = 1;
                for (auto dim : output_shape) {
                    total_output_elements *= dim;
                }
                std::vector<float> output_vector(output_data, output_data + total_output_elements);

                // ����ת��Ϊ uint8 ͼ�񣬲�����
                cv::Mat output_img = postprocess_image(output_vector, output_shape, n_channels);
                std::string output_path = output_dir + "/" + fs::path(img_path).filename().string();
                cv::imwrite(output_path, output_img);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                std::cout << "����ʱ��: " << elapsed.count() << " ms" << std::endl;
                std::cout << "ͼƬ�ѱ�����: " << output_path << std::endl;

                // ����ɹ�������ѭ��
                success = true;
                break;
            }
            catch (const std::exception& ex) {
                std::cerr << "����ͼƬ " << img_path << " ����: " << ex.what() << std::endl;
                retry_count++;

                if (retry_count >= max_retries) {
                    std::cerr << "�ﵽ������Դ��� (" << max_retries << ")��������ͼƬ" << std::endl;
                    break;
                }

                // ���ݵȴ�������
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        } while (!success);
    }

    std::cout << "�ļ��� " << folder_path << " �������" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // ��������
        int noise_level = 40;  // ������׼���� uint8 ͼƬ�߶� 0~255��
        std::string model_name = "ffdnet_gray_clip";  // ģ�����ƣ���������а��� "color"������Ϊ�ǲ�ɫģ��
        int n_channels = (model_name.find("color") != std::string::npos) ? 3 : 1;
        bool use_clip = (model_name.find("clip") != std::string::npos);

        // ���ɴ�ʱ��������Ŀ¼��
        std::string timestamp = get_timestamp();
        std::string base_output_dir = "ȥ��_" + timestamp;

        // ����Ƿ��������в�������ק���ļ��У�
        if (argc > 1) {
            for (int i = 1; i < argc; i++) {
                std::string input_path = argv[i];

                if (fs::exists(input_path)) {
                    if (fs::is_directory(input_path)) {
                        // �����ļ���
                        std::cout << "��ʼ�����ļ���: " << input_path << std::endl;
                        std::string folder_name = fs::path(input_path).filename().string();
                        std::string output_dir = base_output_dir ;
                        process_folder(input_path, output_dir, noise_level, model_name, use_clip, n_channels);
                    }
                    else if (fs::is_regular_file(input_path)) {
                        // �����ļ�ʱҲ������Ӧ���Ŀ¼
                        std::string output_dir = base_output_dir;
                        fs::create_directories(output_dir);

                        // ������ͼƬ�ļ�
                        std::string ext = fs::path(input_path).extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                            std::cout << "��ʼ����ͼƬ: " << input_path << std::endl;

                            // ʹ�� do-while ѭ����֤����ʱ����
                            bool success = false;
                            int retry_count = 0;
                            const int max_retries = 10;

                            do {
                                try {
                                    if (retry_count > 0) {
                                        std::cout << "�� " << retry_count << " �����Դ���ͼƬ: " << input_path << std::endl;
                                    }

                                    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
                                    Ort::Session session{ env, L"ffdnet_gray_clip.onnx", Ort::SessionOptions{ nullptr } };

                                    Ort::AllocatorWithDefaultOptions allocator;
                                    Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
                                    Ort::AllocatedStringPtr sigma_name_Ptr = session.GetInputNameAllocated(1, allocator);

                                    std::vector<int64_t> sigma_shape = { 1, 1, 1, 1 };
                                    std::vector<float> sigma_val(1, noise_level / 255.0f);

                                    size_t num_outputs = session.GetOutputCount();
                                    std::vector<const char*> output_names;
                                    for (size_t i = 0; i < num_outputs; i++) {
                                        output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
                                    }

                                    auto start = std::chrono::high_resolution_clock::now();

                                    std::vector<int64_t> input_shape;
                                    std::vector<float> input_tensor = preprocess_image_vector(input_path, n_channels, 25, use_clip, input_shape);

                                    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                                    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());
                                    Ort::Value sigma_tensor = Ort::Value::CreateTensor<float>(memory_info, sigma_val.data(), sigma_val.size(),
                                        sigma_shape.data(), sigma_shape.size());

                                    const char* input_names_arr[] = { input_name_Ptr.get(), sigma_name_Ptr.get() };
                                    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names_arr,
                                        std::array<Ort::Value, 2>{std::move(input_tensor_ort), std::move(sigma_tensor)}.data(),
                                        2, output_names.data(), output_names.size());

                                    float* output_data = output_tensors.front().GetTensorMutableData<float>();
                                    auto output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
                                    std::vector<int64_t> output_shape = output_info.GetShape();
                                    size_t total_output_elements = 1;
                                    for (auto dim : output_shape) {
                                        total_output_elements *= dim;
                                    }
                                    std::vector<float> output_vector(output_data, output_data + total_output_elements);

                                    cv::Mat output_img = postprocess_image(output_vector, output_shape, n_channels);
                                    std::string output_path = output_dir + "/" + fs::path(input_path).filename().string();
                                    cv::imwrite(output_path, output_img);

                                    auto end = std::chrono::high_resolution_clock::now();
                                    std::chrono::duration<double, std::milli> elapsed = end - start;
                                    std::cout << "����ʱ��: " << elapsed.count() << " ms" << std::endl;
                                    std::cout << "ͼƬ�ѱ�����: " << output_path << std::endl;

                                    success = true;
                                    break;
                                }
                                catch (const std::exception& ex) {
                                    std::cerr << "����ͼƬ " << input_path << " ����: " << ex.what() << std::endl;
                                    retry_count++;

                                    if (retry_count >= max_retries) {
                                        std::cerr << "�ﵽ������Դ��� (" << max_retries << ")��������ͼƬ" << std::endl;
                                        break;
                                    }

                                    // ���ݵȴ�������
                                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                }
                            } while (!success);
                        }
                        else {
                            std::cout << "��֧�ֵ��ļ���ʽ: " << input_path << std::endl;
                        }
                    }
                }
                else {
                    std::cout << "·��������: " << input_path << std::endl;
                }
            }
        }
        else {
            // �������в�����ʹ��Ĭ�ϲ���Ŀ¼
            std::string testset_dir = "examples";  // ����ͼƬ����Ŀ¼
            std::string output_dir = base_output_dir + "/examples";

            if (fs::exists(testset_dir)) {
                std::cout << "δָ�������ļ��У�ʹ��Ĭ�ϲ���Ŀ¼: " << testset_dir << std::endl;
                process_folder(testset_dir, output_dir, noise_level, model_name, use_clip, n_channels);
            }
            else {
                std::cout << "Ĭ�ϲ���Ŀ¼�����ڡ��뽫ͼƬ���� 'examples' �ļ��У���ֱ�ӽ��ļ����Ϸŵ��˳����ϡ�" << std::endl;
                std::cout << "��������˳�..." << std::endl;
                std::cin.get();
            }
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "�������쳣: " << ex.what() << std::endl;
        std::cout << "��������˳�..." << std::endl;
        std::cin.get();
        return -1;
    }
}