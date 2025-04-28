
#include <fstream>

#include "3rdparty/json/json.hpp"
#include "base/map.h"
#include "mapper/incremental_mapper.h"
#include "utility/io_ecim.hpp"
#include "utility/io_feature.hpp"

using namespace xrsfm;

void PreProcess(const std::string dir_path, const std::string camera_path,
                Map &map) {
    std::vector<Frame> frames;
    std::vector<FramePair> frame_pairs;
    ReadFeatures(dir_path + "ftr.bin", frames, true); // 读取特征文件
    ReadFramePairs(dir_path + "fp.bin", frame_pairs); // 读取帧对文件
    std::cout << "ReadFramePairs\n";

    // set cameras & image name
    std::map<int, Camera> cameras = ReadCamerasText(camera_path);
    CHECK_EQ(cameras.size(), 1);                  // 确保只有一个相机
    const int camera_id = cameras.begin()->first; // 获取相机ID

    // 设置每一帧的相机ID
    for (auto &frame : frames) {
        frame.camera_id = camera_id;
    }

    // convert keypoint to points(for reconstruction)
    for (auto &frame : frames) {
        const int num_points = frame.keypoints_.size();
        frame.points.clear();   // 清空之前的点
        frame.track_ids_.assign(num_points, -1);    // 初始化跟踪ID为-1
        for (const auto &kpt : frame.keypoints_) {
            const auto &pt = kpt.pt;
            Eigen::Vector2d ept(pt.x, pt.y), eptn;
            frame.points.push_back(ept);    // 将关键点转换为二维点并添加到帧中
        }
    }

    // 将相机信息、帧和帧对添加到地图中
    map.camera_map_ = cameras;
    map.frames_ = frames;
    map.frame_pairs_ = frame_pairs;

    map.RemoveRedundancyPoints();   // 删除冗余点
    map.Init();                     // 初始化地图
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    // 1.Read Config
    std::string bin_path, camera_path, output_path;
    int init_id1 = -1, init_id2 = -1;
    if (argc <= 2) {
        std::string config_path = "./config_seq.json";
        if (argc == 2) {
            config_path = argv[1];
        }
        auto config_json = LoadJSON(config_path);
        bin_path = config_json["bin_path"];
        camera_path = config_json["camera_path"];
        output_path = config_json["output_path"];
        init_id1 = config_json["init_id1"];
        init_id2 = config_json["init_id2"];
    } else if (argc >= 4 && argc <= 6) {
        bin_path = argv[1];
        camera_path = argv[2];
        output_path = argv[3];
        if (argc >= 5) {
            init_id1 = std::stoi(argv[4]);
        }
        if (argc == 6) {
            init_id2 = std::stoi(argv[5]);
        }
    } else {
        exit(-1);
    }
    std::cout << "Read Config Done!" << std::endl;

    // 2. Map PreProcess
    Map map;
    PreProcess(bin_path, camera_path, map);  // 地图预处理
    std::cout << "PreProcess Done!" << std::endl;

    // 3. Map Reconstruction
    IncrementalMapper imapper;
    imapper.options.init_id1 = init_id1;
    imapper.options.init_id2 = init_id2;
    imapper.options.correct_pose = false;
    imapper.options.stop_when_register_fail = true;
    imapper.Reconstruct(map);
    std::cout << "Reconstruction Done!" << std::endl;

    // 4. Output
    WriteColMapDataBinary(output_path, map);

    return 0;
}
