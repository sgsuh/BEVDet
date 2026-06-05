/*
Create: 2026.06.05
Author: SG.SUH
*/

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "nuscenes_loader.h"
#include "trt_bevdet.h"

namespace {

template <std::size_t N>
void assignFloats(std::unique_ptr<float[]>& dst,
                  const std::array<float, N>& src) {
    dst = std::make_unique<float[]>(N);
    std::copy(src.begin(), src.end(), dst.get());
}

}  // namespace

int main(int, char**) {
    const std::filesystem::path fold_path =
        std::filesystem::current_path().parent_path();
    const std::string bevdet_cfg_path =
        (fold_path / "cfg" / "bevdet.yaml").string();
    const std::string bevdet_param_path =
        (fold_path / "cfg" / "params.yaml").string();

    YAML::Node param_node = YAML::LoadFile(bevdet_param_path);
    auto trt_bevdet = std::make_unique<TRTBEVDet>(param_node);
    trt_bevdet->setConfig(bevdet_cfg_path.c_str());
    if(!trt_bevdet->loadEngine()) {
        trt_bevdet->buildEngine();
    }

    const std::string dataroot =
        param_node["nuscenes_root"].as<std::string>();
    const std::string version =
        param_node["nuscenes_version"].as<std::string>("v1.0-mini");
    const std::size_t max_samples =
        param_node["max_samples"].as<std::size_t>(0);  // 0 = all

    NuscenesLoader loader(dataroot, version);
    const std::size_t total = loader.numFrames();
    const std::size_t n_run =
        max_samples == 0 ? total : std::min(max_samples, total);

    const int num_cam = param_node["num_cam"].as<int>();
    if(num_cam != NUM_CAM) {
        std::cerr << "params.yaml num_cam=" << num_cam
                  << " mismatches NuscenesLoader NUM_CAM=" << NUM_CAM
                  << std::endl;
        return 1;
    }

    for(std::size_t i = 0; i < n_run; ++i) {
        const NuscenesFrame& f = loader.frame(i);

        assignFloats(trt_bevdet->ego2global_rot_, f.ego2global_rot);
        assignFloats(trt_bevdet->ego2global_trans_, f.ego2global_trans);
        assignFloats(trt_bevdet->lidar2ego_rot_, f.lidar2ego_rot);
        assignFloats(trt_bevdet->lidar2ego_trans_, f.lidar2ego_trans);

        std::vector<cv::Mat> imgs(num_cam);
        for(int j = 0; j < num_cam; ++j) {
            imgs[j] = cv::imread(f.cams[j].image_path);
            if(imgs[j].empty()) {
                std::cerr << "Failed to read " << f.cams[j].image_path
                          << std::endl;
                return 1;
            }
            assignFloats(trt_bevdet->cam_intrin_[j], f.cams[j].cam_intrinsic);
            assignFloats(trt_bevdet->sensor2lidar_rot_[j],
                         f.cams[j].sensor2lidar_rot);
            assignFloats(trt_bevdet->sensor2lidar_trans_[j],
                         f.cams[j].sensor2lidar_trans);
        }

        trt_bevdet->frame_index_ = static_cast<int>(i);
        trt_bevdet->vis_imgs_.clear();
        trt_bevdet->vis_imgs_.reserve(imgs.size());
        for(const auto& m : imgs) trt_bevdet->vis_imgs_.push_back(m.clone());
        const auto t0 = std::chrono::steady_clock::now();
        trt_bevdet->inference(imgs);
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "[" << (i + 1) << "/" << n_run << "] "
                  << f.sample_token << "  " << dt << " ms" << std::endl;
    }
    return 0;
}
