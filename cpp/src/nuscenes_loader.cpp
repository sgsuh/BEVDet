/*
Create: 2026.06.05
Author: SG.SUH
*/

#include "nuscenes_loader.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

using json = nlohmann::json;

const std::array<std::string, NUM_CAM> NuscenesLoader::CAM_ORDER = {
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
};

namespace {

json loadJson(const std::string& path) {
    std::ifstream ifs(path);
    if(!ifs) {
        throw std::runtime_error("Failed to open " + path);
    }
    json j;
    ifs >> j;
    return j;
}

inline Eigen::Matrix3f toRot(const json& wxyz) {
    Eigen::Quaternionf q(wxyz[0].get<float>(), wxyz[1].get<float>(),
                         wxyz[2].get<float>(), wxyz[3].get<float>());
    return q.toRotationMatrix();
}

inline Eigen::Vector3f toVec3(const json& v) {
    return {v[0].get<float>(), v[1].get<float>(), v[2].get<float>()};
}

template <std::size_t N>
inline void copyArr(const json& src, std::array<float, N>& dst) {
    for(std::size_t i = 0; i < N; ++i) {
        dst[i] = src[i].get<float>();
    }
}

inline void copyMat(const Eigen::Matrix3f& m, std::array<float, 9>& dst) {
    for(int r = 0; r < 3; ++r) {
        for(int c = 0; c < 3; ++c) {
            dst[r * 3 + c] = m(r, c);
        }
    }
}

inline void copyVec(const Eigen::Vector3f& v, std::array<float, 3>& dst) {
    dst[0] = v.x();
    dst[1] = v.y();
    dst[2] = v.z();
}

}  // namespace

NuscenesLoader::NuscenesLoader(const std::string& dataroot,
                               const std::string& version)
    : dataroot_(dataroot), version_(version) {
    build();
}

void NuscenesLoader::build() {
    const std::string meta_dir = dataroot_ + "/" + version_ + "/";

    auto sample_j           = loadJson(meta_dir + "sample.json");
    auto sample_data_j      = loadJson(meta_dir + "sample_data.json");
    auto ego_pose_j         = loadJson(meta_dir + "ego_pose.json");
    auto calib_sensor_j     = loadJson(meta_dir + "calibrated_sensor.json");
    auto sensor_j           = loadJson(meta_dir + "sensor.json");

    // sensor_token -> channel ("CAM_FRONT", "LIDAR_TOP", ...)
    std::unordered_map<std::string, std::string> sensor_channel;
    for(const auto& s : sensor_j) {
        sensor_channel[s["token"].get<std::string>()] =
            s["channel"].get<std::string>();
    }

    // calibrated_sensor_token -> record
    std::unordered_map<std::string, const json*> calib_by_token;
    for(const auto& c : calib_sensor_j) {
        calib_by_token[c["token"].get<std::string>()] = &c;
    }

    // ego_pose_token -> record
    std::unordered_map<std::string, const json*> ego_by_token;
    for(const auto& e : ego_pose_j) {
        ego_by_token[e["token"].get<std::string>()] = &e;
    }

    // For each keyframe sample, group by channel
    // sample_token -> channel -> sample_data record
    std::unordered_map<std::string,
                       std::unordered_map<std::string, const json*>> by_sample;
    for(const auto& sd : sample_data_j) {
        if(!sd.value("is_key_frame", false)) continue;
        const std::string& cs_tok =
            sd["calibrated_sensor_token"].get_ref<const std::string&>();
        auto it = calib_by_token.find(cs_tok);
        if(it == calib_by_token.end()) continue;
        const std::string& ch =
            sensor_channel[(*it->second)["sensor_token"]
                               .get_ref<const std::string&>()];
        by_sample[sd["sample_token"].get<std::string>()][ch] = &sd;
    }

    frames_.reserve(sample_j.size());

    for(const auto& s : sample_j) {
        const std::string sample_token = s["token"].get<std::string>();
        const auto chmap_it = by_sample.find(sample_token);
        if(chmap_it == by_sample.end()) continue;
        const auto& chmap = chmap_it->second;

        // Require LIDAR_TOP + all 6 cameras present
        auto lidar_it = chmap.find("LIDAR_TOP");
        if(lidar_it == chmap.end()) continue;
        bool ok = true;
        for(const auto& cam : NuscenesLoader::CAM_ORDER) {
            if(chmap.find(cam) == chmap.end()) { ok = false; break; }
        }
        if(!ok) continue;

        NuscenesFrame f;
        f.sample_token = sample_token;
        f.timestamp = s.value("timestamp", std::int64_t{0});

        // Ego pose + lidar2ego from LIDAR_TOP sample_data
        const json& lidar_sd = *lidar_it->second;
        const json& lidar_ego =
            *ego_by_token[lidar_sd["ego_pose_token"]
                              .get_ref<const std::string&>()];
        const json& lidar_calib =
            *calib_by_token[lidar_sd["calibrated_sensor_token"]
                                .get_ref<const std::string&>()];

        copyArr(lidar_ego["rotation"], f.ego2global_rot);
        copyArr(lidar_ego["translation"], f.ego2global_trans);
        copyArr(lidar_calib["rotation"], f.lidar2ego_rot);
        copyArr(lidar_calib["translation"], f.lidar2ego_trans);

        const Eigen::Matrix3f Rl2e = toRot(lidar_calib["rotation"]);
        const Eigen::Vector3f tl2e = toVec3(lidar_calib["translation"]);
        const Eigen::Matrix3f Rl2e_T = Rl2e.transpose();

        for(std::size_t i = 0; i < NUM_CAM; ++i) {
            const json& sd = *chmap.at(NuscenesLoader::CAM_ORDER[i]);
            f.cams[i].image_path =
                dataroot_ + "/" + sd["filename"].get<std::string>();

            const json& cs =
                *calib_by_token[sd["calibrated_sensor_token"]
                                    .get_ref<const std::string&>()];
            // Camera intrinsic comes as 3x3 nested array.
            for(int r = 0; r < 3; ++r) {
                for(int c = 0; c < 3; ++c) {
                    f.cams[i].cam_intrinsic[r * 3 + c] =
                        cs["camera_intrinsic"][r][c].get<float>();
                }
            }
            const Eigen::Matrix3f Rs2e = toRot(cs["rotation"]);
            const Eigen::Vector3f ts2e = toVec3(cs["translation"]);
            // sensor2lidar = inv(lidar2ego) * sensor2ego
            const Eigen::Matrix3f Rs2l = Rl2e_T * Rs2e;
            const Eigen::Vector3f ts2l = Rl2e_T * (ts2e - tl2e);
            copyMat(Rs2l, f.cams[i].sensor2lidar_rot);
            copyVec(ts2l, f.cams[i].sensor2lidar_trans);
        }
        frames_.push_back(std::move(f));
    }

    std::cout << "[NuscenesLoader] " << version_ << ": " << frames_.size()
              << " keyframes from " << dataroot_ << std::endl;
}
