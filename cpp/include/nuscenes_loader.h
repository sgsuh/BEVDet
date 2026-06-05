/*
Create: 2026.06.05
Author: SG.SUH

Minimal nuScenes loader for the C++ TRT BEVDet runner.

Reads the standard v1.0-* JSON layout (sample.json, sample_data.json,
ego_pose.json, calibrated_sensor.json, sensor.json) and exposes one
NuscenesFrame per keyframe with everything the existing inference path
expects: 6 image paths, per-camera intrinsics and sensor2lidar 3x3+t3,
and lidar2ego / ego2global poses as quaternion (w,x,y,z) + translation.
*/

#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

constexpr int NUM_CAM = 6;

struct CamMeta {
    std::string image_path;
    std::array<float, 9> cam_intrinsic;       // row-major 3x3
    std::array<float, 9> sensor2lidar_rot;    // row-major 3x3
    std::array<float, 3> sensor2lidar_trans;
};

struct NuscenesFrame {
    std::string sample_token;
    std::int64_t timestamp;

    std::array<float, 4> ego2global_rot;      // w,x,y,z
    std::array<float, 3> ego2global_trans;
    std::array<float, 4> lidar2ego_rot;       // w,x,y,z
    std::array<float, 3> lidar2ego_trans;

    std::array<CamMeta, NUM_CAM> cams;        // ordered by CAM_ORDER
};

class NuscenesLoader {
public:
    // CAM_ORDER mirrors the original main.cpp:
    //   CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT,
    //   CAM_BACK_LEFT,  CAM_BACK,  CAM_BACK_RIGHT
    static const std::array<std::string, NUM_CAM> CAM_ORDER;

    // dataroot e.g. /workspace/BEVDet/data/nuscenes, version e.g. v1.0-mini.
    NuscenesLoader(const std::string& dataroot, const std::string& version);

    std::size_t numFrames() const { return frames_.size(); }
    const NuscenesFrame& frame(std::size_t i) const { return frames_[i]; }

private:
    void build();

    std::string dataroot_;
    std::string version_;
    std::vector<NuscenesFrame> frames_;
};
