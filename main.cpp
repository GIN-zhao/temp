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

// 预处理：读取图片、归一化、加噪、转换数据格式（HWC->CHW）
// 输入参数：image_path - 图片路径，n_channels - 通道数，noise_level - 噪声标准差（针对 uint8 范围 0~255），use_clip 是否对噪声图像做裁剪，
// 返回值：构造好的输入 tensor 数据，同时更新 input_shape（格式为 {1, C, H, W}）
std::vector<float> preprocess_image_vector(const std::string& image_path, int n_channels, int noise_level, bool use_clip, std::vector<int64_t>& input_shape) {
    cv::Mat img;
    if (n_channels == 1) {
        img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            throw std::runtime_error("Fail to read image " + image_path);
        }
        // 固定大小
        cv::resize(img, img, cv::Size(512, 512));
    }
    else {
        img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Fail to read image " + image_path);
        }
        // 转换 BGR 到 RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    // 转换为 float 并归一化到 [0,1]
    img.convertTo(img, CV_32FC(n_channels), 1.0 / 255.0);

    // 添加高斯噪声：标准差 = noise_level/255.0
    float noise_std = noise_level / 255.0f;
    cv::Mat noise(img.size(), img.type());
    cv::randn(noise, 0, noise_std);
    cv::Mat img_noisy = img + noise;
    if (use_clip) {
        cv::min(img_noisy, 1.0, img_noisy);
        cv::max(img_noisy, 0.0, img_noisy);
    }

    // HWC 转 CHW：先分离通道，再按顺序放入 vector 中
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

// 后处理：将推理输出（tensor 的 float 数组）转换为 cv::Mat 图像，并转换为 uint8 格式保存
// output_shape 格式为 {1, channels, height, width}
cv::Mat postprocess_image(const std::vector<float>& output_tensor, const std::vector<int64_t>& output_shape, int n_channels) {
    int height = static_cast<int>(output_shape[2]);
    int width = static_cast<int>(output_shape[3]);
    int channels = static_cast<int>(output_shape[1]);

    // 将各通道数据复制到 cv::Mat 中
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
    // Clip 到 [0,1] 后转换为 [0,255]
    cv::threshold(merged, merged, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(merged, merged, 0.0, 0.0, cv::THRESH_TOZERO);
    merged = merged * 255.0;
    merged.convertTo(merged, CV_8U);
    if (n_channels != 1) {
        // 将 RGB 转换为 BGR，确保 cv::imwrite 正常保存
        cv::cvtColor(merged, merged, cv::COLOR_RGB2BGR);
    }
    return merged;
}

// 获取当前时间戳字符串，格式为：YYYYMMDD_HHMMSS
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm time_info;
    localtime_s(&time_info, &now_time_t); // Correct usage of localtime_s
    std::stringstream ss;
    ss << std::put_time(&time_info, "%Y%m%d_%H%M%S");
    return ss.str();
}

// 函数用于处理指定文件夹中的图片
void process_folder(const std::string& folder_path, const std::string& output_dir, int noise_level,
    const std::string& model_name, bool use_clip, int n_channels) {
    // 创建输出目录
    fs::create_directories(output_dir);

    // 获取该文件夹下所有图片
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
        std::cout << "目录中未发现图片，请检查路径: " << folder_path << std::endl;
        return;
    }

    // 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

    // 遍历处理每一张图片
    for (const auto& img_path : image_paths) {
        std::cout << "正在处理图片: " << img_path << std::endl;

        // 使用 do-while 循环，确保在出错时重试
        bool success = false;
        int retry_count = 0;
        const int max_retries = 10; // 最大重试次数

        do {
            try {
                if (retry_count > 0) {
                    std::cout << "第 " << retry_count << " 次重试处理图片: " << img_path << std::endl;
                }

                Ort::Session session{ env, L"ffdnet_gray_clip.onnx", Ort::SessionOptions{ nullptr } };

                // 获取输入名称
                Ort::AllocatorWithDefaultOptions allocator;
                Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
                Ort::AllocatedStringPtr sigma_name_Ptr = session.GetInputNameAllocated(1, allocator);

                // 构造 sigma，形状为 (1, 1, 1, 1)
                std::vector<int64_t> sigma_shape = { 1, 1, 1, 1 };
                std::vector<float> sigma_val(1, noise_level / 255.0f);

                // 获取所有输出名称
                size_t num_outputs = session.GetOutputCount();
                std::vector<const char*> output_names;
                for (size_t i = 0; i < num_outputs; i++) {
                    output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
                }

                auto start = std::chrono::high_resolution_clock::now();

                // 预处理：读取、归一化、加噪，并转换为 CHW 格式
                std::vector<int64_t> input_shape;
                std::vector<float> input_tensor = preprocess_image_vector(img_path, n_channels, 25, use_clip, input_shape);

                // 构造输入 tensor 对象
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());
                Ort::Value sigma_tensor = Ort::Value::CreateTensor<float>(memory_info, sigma_val.data(), sigma_val.size(),
                    sigma_shape.data(), sigma_shape.size());

                // 执行推理
                const char* input_names_arr[] = { input_name_Ptr.get(), sigma_name_Ptr.get() };
                auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names_arr,
                    std::array<Ort::Value, 2>{std::move(input_tensor_ort), std::move(sigma_tensor)}.data(),
                    2, output_names.data(), output_names.size());

                // 获取输出 tensor 数据及形状
                float* output_data = output_tensors.front().GetTensorMutableData<float>();
                auto output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
                std::vector<int64_t> output_shape = output_info.GetShape();
                size_t total_output_elements = 1;
                for (auto dim : output_shape) {
                    total_output_elements *= dim;
                }
                std::vector<float> output_vector(output_data, output_data + total_output_elements);

                // 后处理：转换为 uint8 图像，并保存
                cv::Mat output_img = postprocess_image(output_vector, output_shape, n_channels);
                std::string output_path = output_dir + "/" + fs::path(img_path).filename().string();
                cv::imwrite(output_path, output_img);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                std::cout << "处理时间: " << elapsed.count() << " ms" << std::endl;
                std::cout << "图片已保存至: " << output_path << std::endl;

                // 处理成功，跳出循环
                success = true;
                break;
            }
            catch (const std::exception& ex) {
                std::cerr << "处理图片 " << img_path << " 出错: " << ex.what() << std::endl;
                retry_count++;

                if (retry_count >= max_retries) {
                    std::cerr << "达到最大重试次数 (" << max_retries << ")，跳过此图片" << std::endl;
                    break;
                }

                // 短暂等待后重试
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        } while (!success);
    }

    std::cout << "文件夹 " << folder_path << " 处理完毕" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // 参数设置
        int noise_level = 40;  // 噪声标准差（针对 uint8 图片尺度 0~255）
        std::string model_name = "ffdnet_gray_clip";  // 模型名称；如果名称中包含 "color"，则认为是彩色模型
        int n_channels = (model_name.find("color") != std::string::npos) ? 3 : 1;
        bool use_clip = (model_name.find("clip") != std::string::npos);

        // 生成带时间戳的输出目录名
        std::string timestamp = get_timestamp();
        std::string base_output_dir = "去噪_" + timestamp;

        // 检查是否有命令行参数（拖拽的文件夹）
        if (argc > 1) {
            for (int i = 1; i < argc; i++) {
                std::string input_path = argv[i];

                if (fs::exists(input_path)) {
                    if (fs::is_directory(input_path)) {
                        // 处理文件夹
                        std::cout << "开始处理文件夹: " << input_path << std::endl;
                        std::string folder_name = fs::path(input_path).filename().string();
                        std::string output_dir = base_output_dir ;
                        process_folder(input_path, output_dir, noise_level, model_name, use_clip, n_channels);
                    }
                    else if (fs::is_regular_file(input_path)) {
                        // 单个文件时也创建对应输出目录
                        std::string output_dir = base_output_dir;
                        fs::create_directories(output_dir);

                        // 处理单个图片文件
                        std::string ext = fs::path(input_path).extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                            std::cout << "开始处理图片: " << input_path << std::endl;

                            // 使用 do-while 循环保证错误时重试
                            bool success = false;
                            int retry_count = 0;
                            const int max_retries = 10;

                            do {
                                try {
                                    if (retry_count > 0) {
                                        std::cout << "第 " << retry_count << " 次重试处理图片: " << input_path << std::endl;
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
                                    std::cout << "处理时间: " << elapsed.count() << " ms" << std::endl;
                                    std::cout << "图片已保存至: " << output_path << std::endl;

                                    success = true;
                                    break;
                                }
                                catch (const std::exception& ex) {
                                    std::cerr << "处理图片 " << input_path << " 出错: " << ex.what() << std::endl;
                                    retry_count++;

                                    if (retry_count >= max_retries) {
                                        std::cerr << "达到最大重试次数 (" << max_retries << ")，跳过此图片" << std::endl;
                                        break;
                                    }

                                    // 短暂等待后重试
                                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                }
                            } while (!success);
                        }
                        else {
                            std::cout << "不支持的文件格式: " << input_path << std::endl;
                        }
                    }
                }
                else {
                    std::cout << "路径不存在: " << input_path << std::endl;
                }
            }
        }
        else {
            // 无命令行参数，使用默认测试目录
            std::string testset_dir = "examples";  // 测试图片所在目录
            std::string output_dir = base_output_dir + "/examples";

            if (fs::exists(testset_dir)) {
                std::cout << "未指定输入文件夹，使用默认测试目录: " << testset_dir << std::endl;
                process_folder(testset_dir, output_dir, noise_level, model_name, use_clip, n_channels);
            }
            else {
                std::cout << "默认测试目录不存在。请将图片放入 'examples' 文件夹，或直接将文件夹拖放到此程序上。" << std::endl;
                std::cout << "按任意键退出..." << std::endl;
                std::cin.get();
            }
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "程序发生异常: " << ex.what() << std::endl;
        std::cout << "按任意键退出..." << std::endl;
        std::cin.get();
        return -1;
    }
}