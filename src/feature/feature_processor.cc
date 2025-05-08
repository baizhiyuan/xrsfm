
#include "feature_processor.h"

#include <cctype>
#include <experimental/filesystem>
#include <regex>
#include <unordered_set>

#include "base/map.h"
#include "feature/feature_processing.h"
#include "utility/io_ecim.hpp"
#include "utility/timer.h"

namespace xrsfm {

FeatureProcessor::FeatureProcessor(std::string images_path,
                                   std::string output_path,
                                   std::string matching_type,
                                   std::string retrieval_path)
    : images_path_(images_path), output_path_(output_path),
      matching_type_(matching_type), retrieval_path_(retrieval_path) {
    // 检查图像路径是否存在，如果不存在，则输出错误信息并退出
    if (!std::experimental::filesystem::exists(images_path)) {
        std::cout << "image path not exists :" << images_path << "\n";
        exit(-1);
    }
    // 检查输出路径是否存在，如果不存在，则输出错误信息并退出
    if (!std::experimental::filesystem::exists(output_path)) {
        std::cout << "output path not exists :" << output_path << "\n";
        exit(-1);
    }
}

// 该函数用于获取图像的特征和尺寸。
// 如果特征文件和尺寸文件已存在，直接读取文件，否则重新提取图像的特征并保存。
void FeatureProcessor::GetFeatures(const std::string &images_path,
                                   const std::vector<std::string> &image_names,
                                   const std::string &ftr_path,
                                   const std::string &size_path,
                                   std::vector<Frame> &frames,
                                   std::vector<ImageSize> &image_size) {

    // 尝试打开特征文件和尺寸文件
    std::ifstream ftr_bin(ftr_path);
    std::ifstream size_bin(size_path);

    // 如果特征文件和尺寸文件都存在，直接读取文件内容
    if (ftr_bin.good() && size_bin.good()) {
        ReadFeatures(ftr_path, frames);         // 读取特征
        SetUpFramePoints(frames);                       // 设置图像的特征点
        LoadImageSize(size_path, image_size);   // 加载图像尺寸信息
    } else {
        // 如果文件不存在，则从图像路径提取特征并保存
        FeatureExtract(images_path, frames, image_size);    // 提取图像特征
        SaveFeatures(ftr_path, frames, true);                 // 保存提取的特征
        SaveImageSize(size_path, image_size);                          // 保存图像尺寸信息
    }
}

// 该函数用于获取初始的图像帧对。
// 如果已经有保存的帧对文件，它会从文件中读取；如果没有，它会进行特征匹配并保存结果。
void FeatureProcessor::GetInitFramePairs(
    const std::string &path, const std::vector<Frame> &frames,
    const std::vector<std::pair<int, int>> id_pairs,
    std::vector<FramePair> &frame_pairs) {
    // 尝试打开已保存的帧对文件
    std::ifstream bin(path);
    if (bin.good()) {
        // 如果文件存在，读取已有的帧对
        ReadFramePairs(path, frame_pairs);
    } else {
        // 如果文件不存在，进行特征匹配并保存帧对
        FeatureMatching(frames, id_pairs, frame_pairs, true);   // 特征匹配
        SaveFramePairs(path, frame_pairs);  // 保存帧对
    }
}

// 该函数用于根据id2rank（每个图像的检索排名）提取最接近的图像对。
// 它从检索结果中选择num_candidates个最接近的图像，并确保每对图像的顺序（小ID在前）。
// todo: 增加双目匹配的环节
void FeatureProcessor::ExtractNearestImagePairs(
    const std::map<int, std::vector<int>> &id2rank, const int num_candidates,
    std::vector<std::pair<int, int>> &image_id_pairs) {
    std::set<std::pair<int, int>> image_pair_set;
    // search nv5 in candidates except for nearest images
    // 遍历id2rank，提取每个图像的检索结果，选择最接近的图像
    for (const auto &[id, retrieval_results] : id2rank) {
        int count = 0;
        for (const auto &id2 : retrieval_results) {
            // 选择最接近的图像对，确保每对图像不重复
            auto ret = image_pair_set.insert(
                std::make_pair(std::min(id, id2), std::max(id, id2)));
            count++;
            if (count >= num_candidates)
                break;
        }
    }
    // 将结果存入image_id_pairs，并按ID顺序排序
    image_id_pairs.insert(image_id_pairs.end(), image_pair_set.begin(),
                          image_pair_set.end());
    std::sort(image_id_pairs.begin(), image_id_pairs.end(),
              [](auto &a, auto &b) {
                  if (a.first < b.first ||
                      (a.first == b.first && a.second < b.second))
                      return true;
                  return false;
              });
}

// 该函数用于选择两个初始图像ID，通常是通过计算图像之间的连接度（即匹配数）来确定最相关的两张图像。
// todo：对应的左右图像，前后两帧应该是最相关的，需要改进
std::tuple<int, int>
FeatureProcessor::GetInitId(const int num_image,
                            std::vector<FramePair> &frame_pairs) {
    std::vector<std::map<int, int>> id2cor_num_vec(num_image);
    std::vector<std::pair<int, int>> connect_number_vec(num_image);
    std::map<int, std::set<int>> connect_id_vec;

    // 初始化连接度为0
    for (int i = 0; i < num_image; ++i) {
        connect_number_vec[i] = {i, 0};
    }

    // 计算每一对图像的连接度, 只考虑匹配点数大于100的帧对
    for (auto &fp : frame_pairs) {
        if (fp.inlier_num < 100)
            continue; // for Trafalgar
        connect_number_vec[fp.id1].second++;
        connect_number_vec[fp.id2].second++;
        id2cor_num_vec[fp.id1][fp.id2] = fp.matches.size();
        id2cor_num_vec[fp.id2][fp.id1] = fp.matches.size();
    }
    // 按照连接度排序，选出最相关的两张图像
    std::sort(connect_number_vec.begin(), connect_number_vec.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                  return a.second > b.second;
              });
    const int init_id1 = connect_number_vec[0].first;
    int init_id2 = -1;
    // 找到与初始图像ID连接度最强的另一张图像
    for (auto &[id, number] : connect_number_vec) {
        if (id2cor_num_vec[init_id1].count(id) != 0 &&
            id2cor_num_vec[init_id1][id] >= 100) {
            init_id2 = id;
            break;
        }
    }
    return std::tuple<int, int>(init_id1, init_id2);    // 返回初始ID对
}

// 该函数执行顺序匹配，首先将相邻的图像帧配对，然后基于检索结果进行匹配，最终将所有匹配对存储在id_pairs中
// ToDo: 需要增加双目之间的匹配
void FeatureProcessor::MatchingSeq(
    std::vector<Frame> &frames, const std::string &fp_path,
    const std::map<int, std::vector<int>> &id2rank,
    std::vector<std::pair<int, int>> &id_pairs) {
    const int num_frame = frames.size();
    std::set<std::pair<int, int>> set_pairs;

    // 遍历帧，按顺序匹配相邻帧
    for (int i = 0; i < num_frame; ++i) {
        for (int k = 1; k < 20 && i + k < num_frame; ++k)
            set_pairs.insert(std::pair<int, int>(i, i + k));
    }

    // 使用检索结果进行匹配
    for (const auto &[id1, vec] : id2rank) {
        if (id1 % 5 != 0)
            continue;
        for (auto &id2 : vec) {
            if (id1 < id2)
                set_pairs.insert(std::pair<int, int>(id1, id2));
            else if (id2 < id1)
                set_pairs.insert(std::pair<int, int>(id2, id1));
        }
    }

    // 将匹配对存入id_pairs并返回
    id_pairs.assign(set_pairs.begin(), set_pairs.end());
}

void FeatureProcessor::Run() {
    const std::string ftr_path = output_path_ + "ftr.bin";
    const std::string size_path = output_path_ + "size.bin";
    const std::string fp_init_path = output_path_ + "fp_init.bin";
    const std::string fp_path = output_path_ + "fp.bin";

    // 2.read images
    std::vector<std::string> image_names;
    LoadImageNames(images_path_, image_names);
    const int num_image = image_names.size();
    std::cout << "Load Image Info Done.\n";

    // 初始化帧信息
    std::vector<Frame> frames;
    frames.resize(num_image);
    // 填充每个帧的ID和名称
    for (int i = 0; i < num_image; ++i) {
        frames[i].id = i;
        frames[i].name = image_names[i];
    }

    // 2.feature extraction
    std::vector<ImageSize> image_size_vec;
    GetFeatures(images_path_, image_names, ftr_path, size_path, frames,
                image_size_vec);
    std::cout << "Extract Features Done.\n";

    // 3.image matching
    std::map<int, std::vector<int>> id2rank;
    std::map<std::string, int> name2id;
    for (int i = 0; i < num_image; ++i) {
        name2id[image_names[i]] = i;
    }
    // 检查是否有检索信息文件，若有则加载，否则进行错误提示
    bool have_retrieval_info =
        LoadRetrievalRank(retrieval_path_, name2id, id2rank);
    std::cout << "Load Retrieval Info Done.\n";

    // 初始化计时器
    Timer timer("%lf s\n");
    timer.start();

    // 根据匹配类型进行图像匹配
    std::vector<FramePair> frame_pairs;
    if (matching_type_ == "covisibility") {
        if (!have_retrieval_info) {
            std::cout << "The retrieval file is required when using "
                         "covisibility-based matching!!!\n";
            return 0;
        }

        // 提取最近的图像对
        std::vector<std::pair<int, int>> id_pairs;
        ExtractNearestImagePairs(id2rank, 5, id_pairs);
        // 获取初始的图像帧对
        GetInitFramePairs(fp_init_path, frames, id_pairs, frame_pairs);
        std::cout << "Init Matching Done.\n";

        // 设置匹配的迭代次数和使用基础矩阵进行匹配的标志
        constexpr int num_iteration = 5;        // 设置迭代次数
        constexpr bool use_fundamental = true;  // 是否使用基础矩阵
        // 获取初始图像对的ID
        const auto [init_id1, init_id2] = GetInitId(num_image, frame_pairs);

        // 创建Map对象，用于存储图像和帧对信息
        Map map;
        map.frames_ = std::move(frames);            // 移动帧数据到Map对象
        map.frame_pairs_ = std::move(frame_pairs);  // 移动帧对数据到Map对象
        // 执行图像匹配和扩展操作
        ExpansionAndMatching(map, id2rank, num_iteration, image_size_vec,
                             init_id1, init_id2, use_fundamental, id_pairs);
    } else {
        // 如果匹配类型不是共视匹配
        std::vector<std::pair<int, int>> id_pairs;
        if (matching_type_ == "sequential") {
            // 如果匹配类型是顺序匹配
            if (!have_retrieval_info) {
                // 如果没有检索信息文件，顺序匹配仅能匹配相邻帧
                std::cout << "Without the retrieval file, sequential matching "
                             "method only matches adjacent frames.\n";
            }
            // 执行顺序匹配
            MatchingSeq(frames, fp_path, id2rank, id_pairs);
        } else if (matching_type_ == "retrieval") {
            // 如果匹配类型是基于检索的匹配
            if (!have_retrieval_info) {
                // 如果没有检索信息文件，输出错误并退出
                std::cout << "ERROR: The retrieval file is required when using "
                             "retrieval-based matching!!!\n";
                return 0;
            }
            // 提取最近的图像对
            ExtractNearestImagePairs(id2rank, 25, id_pairs); // 提取25个最近的图像对
        }
        // 执行特征匹配
        FeatureMatching(frames, id_pairs, frame_pairs, true); // 执行特征匹配
    }
    // 保存帧对到文件
    SaveFramePairs(fp_path, frame_pairs);

    // 停止计时器并输出耗时
    timer.stop();
    timer.print();
}

} // namespace xrsfm
