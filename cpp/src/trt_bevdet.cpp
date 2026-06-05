/*
Create: 2026.05.31
Author: SG.SUH
*/

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "trt_bevdet.h"

TRTBEVDet::TRTBEVDet(YAML::Node& node) {
    init_ = false;
    allocMeta(node["ranks_bev_path"].as<std::string>().c_str(), ranks_bev_);
    allocMeta(node["ranks_depth_path"].as<std::string>().c_str(), ranks_depth_);
    allocMeta(node["ranks_feat_path"].as<std::string>().c_str(), ranks_feat_);
    allocMeta(node["interval_starts_path"].as<std::string>().c_str(), interval_starts_);
    allocMeta(node["interval_lengths_path"].as<std::string>().c_str(), interval_lengths_);
    num_cam_ = node["num_cam"].as<int>();
    resize_width_ = node["resize_width"].as<int>();
    resize_height_ = node["resize_height"].as<int>();
    crop_left_ = node["crop_left"].as<int>();
    crop_right_ = node["crop_right"].as<int>();
    crop_top_ = node["crop_top"].as<int>();
    crop_bottom_ = node["crop_bottom"].as<int>();
    pre_max_size_ = node["pre_max_size"].as<int>();
    post_max_size_ = node["post_max_size"].as<int>();
    max_num_ = node["max_num"].as<int>();
    code_size_ = node["code_size"].as<int>();
    out_size_factor_ = node["out_size_factor"].as<int>();
    openmp_num_threads_ = node["openmp_num_threads"].as<int>();
    score_threshold_ = node["score_threshold"].as<float>();
    num_class_ = node["num_class"].as<std::vector<int>>();
    num_task_ = static_cast<int>(num_class_.size());
    nms_type_ = node["nms_type"].as<std::vector<int>>();
    min_radius_ = node["min_radius"].as<std::vector<float>>();
    voxel_size_ = node["voxel_size"].as<std::vector<float>>();
    pc_range_ = node["pc_range"].as<std::vector<float>>();
    post_center_range_ = node["post_center_range"].as<std::vector<float>>();
    nms_thresh_ = node["nms_thresh"].as<std::vector<float>>();
    nms_rescale_factor_ = node["nms_rescale_factor"].as<std::vector<std::vector<float>>>();
    draw_boxes_indexes_ = node["draw_boxes_indexes"].as<std::vector<std::vector<int>>>();
    draw_boxes_indexes_bev_ = node["draw_boxes_indexes_bev"].as<std::vector<std::vector<int>>>();
    color_map_ = node["color_map"].as<std::vector<std::vector<int>>>();
    vis_ = node["vis"].as<bool>();
    vis_output_dir_ = node["vis_output_dir"].as<std::string>("work_dirs/cpp_vis");
    mean_ = node["mean"].as<std::vector<float>>();
    std_ = node["std"].as<std::vector<float>>();
    corners_norm_ = node["corners_norm"].as<std::vector<std::vector<float>>>();
    sensor2lidar_rot_.resize(num_cam_);
    sensor2lidar_trans_.resize(num_cam_);
    cam_intrin_.resize(num_cam_);

    for(int i = 0; i < stream_.size(); ++i) {
        cudaStreamCreate(&stream_[i]);
    }
}

TRTBEVDet::~TRTBEVDet() {
    for(int i = 0; i < stream_.size(); ++i) {
        cudaStreamDestroy(stream_[i]);
    }
}

void TRTBEVDet::allocMeta(const char* path, std::unique_ptr<int[]>& data) {
    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(0, ifs.end);
    int len = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    data = std::make_unique<int[]>(len / sizeof(int));
    ifs.read((char*)data.get(), len);
    ifs.close();
}

void TRTBEVDet::allocMat(const char* path, std::unique_ptr<float[]>& data) {
    data.reset();
    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(0, ifs.end);
    int len = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    data = std::make_unique<float[]>(len / sizeof(float));
    ifs.read((char*)data.get(), len);
    ifs.close();
}

void TRTBEVDet::tensorFromImg(std::vector<cv::Mat> imgs) {
    #pragma omp parallel for num_threads(openmp_num_threads_)
    for(int i = 0; i < imgs.size(); ++i) {
        const long idx0 = i * dim_in_[0].d[1] * dim_in_[0].d[2] * dim_in_[0].d[3];

        for(int c = 0; c < dim_in_[0].d[1]; ++c) {
            const long idx1 = idx0 + c * dim_in_[0].d[2] * dim_in_[0].d[3];

            for(int h = 0; h < dim_in_[0].d[2]; ++h) {
                const long idx2 = idx1 + h * dim_in_[0].d[3];

                for(int w = 0; w < dim_in_[0].d[3]; ++w) {
                    const long idx3 = idx2 + w;
                    tensor_in_[idx3] = (float)imgs[i].at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }
}

void TRTBEVDet::preprocess(std::vector<cv::Mat>& imgs) {
    if(!init_) {
        int dim_size = 1;

        for(int i = 0; i < dim_in_[0].nbDims; ++i) {
            dim_size *= dim_in_[0].d[i];
        }

        tensor_in_ = std::make_unique<float[]>(dim_size);

        // result_serialize layout: per-task [reg, height, dim, rot, vel, heatmap]
        auto bufSize = [](const nvinfer1::Dims& d) {
            int n = 1;
            for(int k = 0; k < d.nbDims; ++k) n *= d.d[k];
            return n;
        };
        for(int t = 0; t < num_task_; ++t) {
            const int base = t * 6;
            reg_.push_back(std::make_unique<float[]>(bufSize(dim_out_[base + 0])));
            height_.push_back(std::make_unique<float[]>(bufSize(dim_out_[base + 1])));
            dim_.push_back(std::make_unique<float[]>(bufSize(dim_out_[base + 2])));
            rot_.push_back(std::make_unique<float[]>(bufSize(dim_out_[base + 3])));
            vel_.push_back(std::make_unique<float[]>(bufSize(dim_out_[base + 4])));
            heatmap_.push_back(std::make_unique<float[]>(bufSize(dim_out_[base + 5])));
        }

        init_ = true;
    }

    #pragma omp parallel for num_threads(openmp_num_threads_)
    for(int i = 0; i < imgs.size(); ++i) {
        cv::resize(imgs[i], imgs[i], cv::Size(resize_width_, resize_height_));
        imgs[i] = imgs[i](cv::Rect(crop_left_, crop_top_, (crop_right_ - crop_left_), (crop_bottom_ - crop_top_)));
        imgs[i].convertTo(imgs[i], CV_32FC3);

        for(cv::MatIterator_<cv::Vec3f> it = imgs[i].begin<cv::Vec3f>(), end = imgs[i].end<cv::Vec3f>(); it != end; ++it) {
            for(int j = 0; j < 3; ++j) {
                (*it)[j] -= mean_[j];
                (*it)[j] /= std_[j];
            }
        }
    }

    tensorFromImg(imgs);
}

bool cmp(std::pair<int, float>& a, std::pair<int, float>& b) {
    if(a.second == b.second) {
        return a.first > b.first;
    }

    return a.second > b.second;
}

void TRTBEVDet::topk(std::vector<std::pair<int, float>>& heat, 
                    int k, 
                    int task_id,
                    float* scores,
                    float* inds,
                    float* clses,
                    float* ys,
                    float* xs) {
    const int height = dim_out_[task_id * 6 + 5].d[2];
    const int width = dim_out_[task_id * 6 + 5].d[3];
    std::sort(heat.begin(), heat.end(), cmp);

    for(int i = 0; i < max_num_; ++i) {
        scores[i] = heat[i].second;
        inds[i] = heat[i].first % (height * width);
        ys[i] = float(int(inds[i]) / width);
        xs[i] = float(int(inds[i]) % width);

        // 1-task-10-class head: heat is flattened over [C, H, W], so the
        // channel index (flat_idx / (H*W)) is the class label directly (0..C-1).
        // This generalizes the old 2-class-per-task split.
        clses[i] = heat[i].first / (height * width);
    }
}

void TRTBEVDet::decode(std::vector<std::pair<int, float>>& heat,
                        float* rot,
                        float* hei,
                        float* dim,
                        float* vel,
                        float* reg,
                        int task_id,
                        std::vector<std::vector<float>>& final_bboxes,
                        std::vector<float>& final_scores,
                        std::vector<float>& final_labels) {
    float scores[max_num_];
    float inds[max_num_];
    float clses[max_num_];
    float ys[max_num_];
    float xs[max_num_];
    float atan[max_num_];
    topk(heat, max_num_, task_id, scores, inds, clses, ys, xs);
    const int height = dim_out_[task_id * 6 + 5].d[2];
    const int width = dim_out_[task_id * 6 + 5].d[3];

    for(int i = 0; i < max_num_; ++i) {
        const int x_idx = int(inds[i]);
        const int y_idx = int(inds[i]) + height * width;
        const int z_idx = int(inds[i]) + height * width * 2;
        xs[i] += reg[x_idx];
        ys[i] += reg[y_idx];
        atan[i] = std::atan2(rot[x_idx], rot[y_idx]);
        xs[i] = xs[i] * out_size_factor_ * voxel_size_[0] + pc_range_[0];
        ys[i] = ys[i] * out_size_factor_ * voxel_size_[1] + pc_range_[1];

        if(xs[i] < post_center_range_[0] || ys[i] < post_center_range_[1] || hei[x_idx] < post_center_range_[2]) {
            continue;
        }

        if(xs[i] > post_center_range_[3] || ys[i] > post_center_range_[4] || hei[x_idx] > post_center_range_[5]) {
            continue;
        }

        if(scores[i] <= score_threshold_) {
            continue;
        }

        // norm_bbox=True: dim is regressed in log space, so exponentiate to get
        // real-world box extents (matches torch.exp(dim) in CenterHead.get_bboxes).
        final_bboxes.push_back({xs[i], ys[i], hei[x_idx], std::exp(dim[x_idx]), std::exp(dim[y_idx]), std::exp(dim[z_idx]), atan[i], vel[x_idx], vel[y_idx]});
        final_scores.push_back(scores[i]);
        final_labels.push_back(clses[i]);
    }
}

void TRTBEVDet::xywhr2xyxyr(std::vector<std::vector<float>>& boxes_xywhr, std::vector<std::vector<float>>& boxes_xyxyr) {
    for(int i = 0; i < boxes_xywhr.size(); ++i) {
        std::vector<float> boxes;
        boxes.reserve(5);
        float half_w = boxes_xywhr[i][3] / 2.0;
        float half_h = boxes_xywhr[i][4] / 2.0;
        boxes[0] = boxes_xywhr[i][0] - half_w;
        boxes[1] = boxes_xywhr[i][1] - half_h;
        boxes[2] = boxes_xywhr[i][0] + half_w;
        boxes[3] = boxes_xywhr[i][1] + half_h;
        boxes[4] = boxes_xywhr[i][6];
        boxes_xyxyr.push_back(boxes);
    }
}

void TRTBEVDet::getTaskDetections(int num_class_with_bg,
                                std::vector<float>& cls_preds,
                                std::vector<std::vector<float>>& reg_preds,
                                std::vector<float>& cls_labels,
                                int task_id) {
    for(int i = 0; i < cls_preds.size(); ++i) {
        if(nms_rescale_factor_[task_id].size() > 1) {
            for(int j = 0; j < nms_rescale_factor_[task_id].size(); ++j) {
                if(cls_labels[i] == j) {
                    reg_preds[i][3] *= nms_rescale_factor_[task_id][j];
                    reg_preds[i][4] *= nms_rescale_factor_[task_id][j];
                    reg_preds[i][5] *= nms_rescale_factor_[task_id][j];
                }
            }
        } else {
            reg_preds[i][3] *= nms_rescale_factor_[task_id][0];
            reg_preds[i][4] *= nms_rescale_factor_[task_id][0];
            reg_preds[i][5] *= nms_rescale_factor_[task_id][0];
        }
    }

    std::vector<std::vector<float>> boxes_for_nms;
    xywhr2xyxyr(reg_preds, boxes_for_nms);

    for(int i = 0; i < cls_preds.size(); ++i) {
        if(nms_rescale_factor_[task_id].size() > 1) {
            for(int j = 0; j < nms_rescale_factor_[task_id].size(); ++j) {
                if(cls_labels[i] == j) {
                    reg_preds[i][3] /= nms_rescale_factor_[task_id][j];
                    reg_preds[i][4] /= nms_rescale_factor_[task_id][j];
                    reg_preds[i][5] /= nms_rescale_factor_[task_id][j];
                }
            }
        } else {
            reg_preds[i][3] /= nms_rescale_factor_[task_id][0];
            reg_preds[i][4] /= nms_rescale_factor_[task_id][0];
            reg_preds[i][5] /= nms_rescale_factor_[task_id][0];
        }
    }
}

std::vector<int> TRTBEVDet::circleNms(std::vector<std::vector<float>>& final_bboxes,
                                    const float thresh,
                                    const int post_max_size) {
    std::vector<int> keep;
    std::vector<int> suppressed;
    suppressed.resize(final_bboxes.size());

    for(int i = 0; i < final_bboxes.size(); ++i) {
        if(keep.size() >= post_max_size) {
            break;
        }

        if(suppressed[i] == 1) {
            continue;
        }

        keep.emplace_back(i);

        for(int j = 1; j < final_bboxes.size(); ++j) {
            if(suppressed[j] == 1) {
                continue;
            }

            const float dist = (final_bboxes[i][0] - final_bboxes[j][0]) * (final_bboxes[i][0] - final_bboxes[j][0]) + (final_bboxes[i][1] - final_bboxes[j][1]) * (final_bboxes[i][1] - final_bboxes[j][1]);

            if(dist <= thresh) {
                suppressed[j] = 1;
            }
        }
    }

    return keep;
}

void TRTBEVDet::getBboxes(std::vector<std::vector<std::vector<float>>>& boxes_3d,
                        std::vector<std::vector<float>>& scores_3d,
                        std::vector<std::vector<float>>& labels_3d) {
    for(int i = 0; i < num_task_; ++i) {
        std::vector<std::pair<int, float>> heatmap;
        float max_val = 0.0;

        // The exported heatmap is raw logits (CenterHead applies sigmoid in
        // get_bboxes, which is not part of the TRT forward). Apply it here so
        // score_threshold_ is compared in probability space.
        for(int j = 0; j < size_out_[i * 6 + 5] / sizeof(float); ++j) {
            heatmap.push_back({j, 1.0f / (1.0f + std::exp(-heatmap_[i][j]))});
        }

        float* reg = reg_[i].get();
        float* hei = height_[i].get();
        float* dim = dim_[i].get();
        float* rot = rot_[i].get();
        float* vel = vel_[i].get();
        std::vector<std::vector<float>> final_bboxes;
        std::vector<float> final_scores;
        std::vector<float> final_labels;
        decode(heatmap, rot, hei, dim, vel, reg, i, final_bboxes, final_scores, final_labels);

        if(nms_type_[i] == 0) {
            if(final_bboxes.size() == 0) {
                boxes_3d.push_back(final_bboxes);
                scores_3d.push_back(final_scores);
                labels_3d.push_back(final_labels);

                continue;
            } else {
                std::vector<std::vector<float>> temp_final_bboxes;
                std::vector<float> temp_final_scores;
                std::vector<float> temp_final_labels;
                std::vector<int> keep = circleNms(final_bboxes, min_radius_[i], post_max_size_);

                for(int j = 0; j < keep.size(); ++j) {
                    temp_final_bboxes.push_back(final_bboxes[keep[j]]);
                    temp_final_scores.push_back(final_scores[keep[j]]);
                    temp_final_labels.push_back(final_labels[keep[j]]);
                }

                for(int j = 0; j < temp_final_bboxes.size(); ++j) {
                    temp_final_bboxes[j][2] = temp_final_bboxes[j][2] - temp_final_bboxes[j][5] * 0.5;
                }

                boxes_3d.push_back(temp_final_bboxes);
                scores_3d.push_back(temp_final_scores);
                labels_3d.push_back(temp_final_labels);
            }
        } else {
            if(final_bboxes.size() == 0) {
                boxes_3d.push_back(final_bboxes);
                scores_3d.push_back(final_scores);
                labels_3d.push_back(final_labels);

                continue;
            }
        }
    }
}

void TRTBEVDet::postprocess(std::vector<cv::Mat> imgs) {
    std::vector<std::vector<std::vector<float>>> boxes_3d;
    std::vector<std::vector<float>> scores_3d;
    std::vector<std::vector<float>> labels_3d;
    getBboxes(boxes_3d, scores_3d, labels_3d);

    if(vis_) {
        // Prefer the original (un-resized, CV_8UC3) frames if main set them; otherwise fall back to the
        // preprocessed in-place imgs (which will be 256x704 CV_32FC3 and produce a degraded canvas).
        std::vector<cv::Mat>& vimgs = vis_imgs_.empty() ? imgs : vis_imgs_;
        const int canvas_size = 3000;
        cv::Mat canvas = cv::Mat::zeros(canvas_size, canvas_size, CV_8UC3);

        for(int sample_id = 0; sample_id < boxes_3d.size(); ++sample_id) {
            std::vector<std::vector<float>> boxes = boxes_3d[sample_id];
            std::vector<float> scores = scores_3d[sample_id];
            std::vector<float> labels = labels_3d[sample_id];
            Eigen::Quaternionf quat_r(ego2global_rot_[0], ego2global_rot_[1], ego2global_rot_[2], ego2global_rot_[3]);
            Eigen::Quaternionf quat_lidar2ego(lidar2ego_rot_[0], lidar2ego_rot_[1], lidar2ego_rot_[2], lidar2ego_rot_[3]);
            Eigen::Matrix3f rot_r = quat_r.toRotationMatrix();
            Eigen::Matrix3f rot_lidar2ego = quat_lidar2ego.toRotationMatrix();

            for(int i = 0; i < boxes.size(); ++i) {
                std::vector<float> box = boxes[i];
                std::vector<float> center = {box[0], box[1], box[2]};
                std::vector<float> wlh = {box[4], box[3], box[5]};
                float box_yaw = box[6];
                std::vector<float> box_vel = {box[7], box[8], 0.0f};
                Eigen::Quaternionf quat_q(std::cos(box_yaw / 2), 0, 0, std::sin(box_yaw / 2));
                std::vector<float> center_rot;
                std::vector<float> vel_rot;
                center_rot.reserve(3);
                vel_rot.reserve(3);
                Eigen::Quaternionf quat_r_q = quat_r * quat_q;

                for(int j = 0; j < 3; ++j) {
                    center_rot[j] = rot_r(j, 0) * center[0] + rot_r(j, 1) * center[1] + rot_r(j, 2) * center[2] + ego2global_trans_[j];
                    vel_rot[j] = rot_r(j, 0) * box_vel[0] + rot_r(j, 1) * box_vel[1] + rot_r(j, 2) * box_vel[2];
                }

                float quad_yaw = std::atan2(2 * (quat_r_q.w() * quat_r_q.z() + quat_r_q.x() * quat_r_q.y()), 1 - 2 * (quat_r_q.y() * quat_r_q.y() + quat_r_q.z() * quat_r_q.z()));
                quad_yaw = -quad_yaw - CV_PI / 2;
                center_rot[2] += wlh[2] * (-0.5);
                Eigen::MatrixXd corners(8, 3);

                for(int j = 0; j < corners_norm_.size(); ++j) {
                    for(int k = 0; k < corners_norm_[0].size(); ++k) {
                        corners(j, k) = wlh[k] * corners_norm_[j][k];
                    }
                }

                float rot_sin = std::sin(quad_yaw);
                float rot_cos = std::cos(quad_yaw);
                Eigen::Matrix3d rot_mat_t = Eigen::Matrix3d::Identity();
                rot_mat_t(0, 0) = rot_cos;
                rot_mat_t(0, 1) = -rot_sin;
                rot_mat_t(1, 0) = rot_sin;
                rot_mat_t(1, 1) = rot_cos;
                corners = corners * rot_mat_t;
                Eigen::MatrixXd corners_concat(8, 4);

                for(int j = 0; j < corners.rows(); ++j) {
                    for(int k = 0; k < corners.cols(); ++k) {
                        corners_concat(j, k) = corners(j, k) + center_rot[k];
                    }

                    corners_concat(j, 3) = 1.0f;
                }

                Eigen::Matrix4d lidar2ego = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d ego2global = Eigen::Matrix4d::Identity();
                lidar2ego.block<3, 3>(0, 0) = rot_lidar2ego.cast<double>();
                lidar2ego.block<3, 1>(0, 3) = Eigen::Map<const Eigen::Vector3f>(lidar2ego_trans_.get()).cast<double>();
                ego2global.block<3, 3>(0, 0) = rot_r.cast<double>();
                ego2global.block<3, 1>(0, 3) = Eigen::Map<const Eigen::Vector3f>(ego2global_trans_.get()).cast<double>();
                Eigen::Matrix4d l2g = ego2global * lidar2ego;
                Eigen::MatrixXd corners_lidar(8, 4);
                corners_lidar = corners_concat * l2g.inverse().transpose();
                int label = int(labels[i]);

                for(int view = 0; view < num_cam_; ++view) {
                    Eigen::Matrix4d camera2lidar = Eigen::Matrix4d::Identity();

                    for(int j = 0; j < camera2lidar.rows() - 1; ++j) {
                        for(int k = 0; k < camera2lidar.cols() - 1; ++k) {
                            camera2lidar(j, k) = sensor2lidar_rot_[view][j * 3 + k];
                        }

                        camera2lidar(j, 3) = sensor2lidar_trans_[view][j];
                    }

                    Eigen::Matrix4d lidar2camera = camera2lidar.inverse();
                    Eigen::MatrixXd points_camera_homogeneous(8, 4);
                    points_camera_homogeneous = corners_lidar * lidar2camera.transpose();
                    bool valid = true;
                    Eigen::MatrixXd points_camera(8, 3);
                    points_camera = points_camera_homogeneous.block<8, 3>(0, 0);

                    for(int j = 0; j < points_camera.rows(); ++j) {
                        if(points_camera(j, 2) <= 0.5f) {
                            valid = false;

                            break;
                        }

                        points_camera.block<1, 3>(j, 0) /= points_camera(j, 2);
                    }

                    if(valid == false) {
                        continue;
                    }

                    Eigen::Matrix3d camera2img;

                    for(int j = 0; j < camera2img.rows(); ++j) {
                        for(int k = 0; k < camera2img.cols(); ++k) {
                            camera2img(j, k) = cam_intrin_[view][j * 3 + k];
                        }
                    }

                    Eigen::MatrixXd points_img(8, 2);
                    points_img = (points_camera * camera2img.transpose()).block<8, 2>(0, 0);
                    std::array<bool, 8> valid_pts;

                    for(int j = 0; j < points_img.rows(); ++j) {
                        const bool is_valid = points_img(j, 0) >= 0 && points_img(j, 0) < vimgs[view].cols && points_img(j, 1) >= 0 && points_img(j, 1) < vimgs[view].rows;
                        valid_pts[j] = is_valid;
                    }

                    const int color_idx = label % static_cast<int>(color_map_.size());
                    const cv::Scalar edge_color(color_map_[color_idx][0], color_map_[color_idx][1], color_map_[color_idx][2]);
                    for(int j = 0; j < draw_boxes_indexes_.size(); ++j) {
                        const int va = draw_boxes_indexes_[j][0];
                        const int vb = draw_boxes_indexes_[j][1];
                        if(valid_pts[va] && valid_pts[vb]) {
                            const int x1 = static_cast<int>(points_img(va, 0));
                            const int y1 = static_cast<int>(points_img(va, 1));
                            const int x2 = static_cast<int>(points_img(vb, 0));
                            const int y2 = static_cast<int>(points_img(vb, 1));
                            cv::line(vimgs[view], cv::Point(x1, y1), cv::Point(x2, y2), edge_color, 2);
                        }
                    }
                }

                Eigen::MatrixXi bottom_corners_bev(4, 2);
                bottom_corners_bev(0, 0) = std::round((corners_lidar(0, 0) + 50) / 50 / 2.0 * canvas_size);
                bottom_corners_bev(0, 1) = std::round((-corners_lidar(0, 1) + 50) / 50 / 2.0 * canvas_size);
                bottom_corners_bev(1, 0) = std::round((corners_lidar(3, 0) + 50) / 50 / 2.0 * canvas_size);
                bottom_corners_bev(1, 1) = std::round((-corners_lidar(3, 1) + 50) / 50 / 2.0 * canvas_size);
                bottom_corners_bev(2, 0) = std::round((corners_lidar(7, 0) + 50) / 50 / 2.0 * canvas_size);
                bottom_corners_bev(2, 1) = std::round((-corners_lidar(7, 1) + 50) / 50 / 2.0 * canvas_size);
                bottom_corners_bev(3, 0) = std::round((corners_lidar(4, 0) + 50) / 50 / 2.0 * canvas_size);
                bottom_corners_bev(3, 1) = std::round((-corners_lidar(4, 1) + 50) / 50 / 2.0 * canvas_size);
                std::array<float, 2> center_bev;
                center_bev[0] = (corners_lidar(0, 0) + corners_lidar(3, 0) + corners_lidar(7, 0) + corners_lidar(4, 0)) / 4.0f;
                center_bev[1] = -(corners_lidar(0, 1) + corners_lidar(3, 1) + corners_lidar(7, 1) + corners_lidar(4, 1)) / 4.0f;
                std::array<float, 2> head_bev;
                head_bev[0] = (corners_lidar(0, 0) + corners_lidar(4, 0)) / 2.0f;
                head_bev[1] = -(corners_lidar(0, 1) + corners_lidar(4, 1)) / 2.0f;
                std::array<int, 2> center_canvas;
                std::array<int, 2> head_canvas;

                for(int j = 0; j < center_canvas.size(); ++j) {
                    center_canvas[j] = std::round((center_bev[j] + 50) / 50.0f / 2.0f * canvas_size);
                    head_canvas[j] = std::round((head_bev[j] + 50) / 50.0f / 2.0f * canvas_size);
                }

                const int bev_color_idx = label % static_cast<int>(color_map_.size());
                const cv::Scalar bev_color(color_map_[bev_color_idx][0], color_map_[bev_color_idx][1], color_map_[bev_color_idx][2]);
                for(int j = 0; j < draw_boxes_indexes_bev_.size(); ++j) {
                    const int x1 = bottom_corners_bev(draw_boxes_indexes_bev_[j][0], 0);
                    const int y1 = bottom_corners_bev(draw_boxes_indexes_bev_[j][0], 1);
                    const int x2 = bottom_corners_bev(draw_boxes_indexes_bev_[j][1], 0);
                    const int y2 = bottom_corners_bev(draw_boxes_indexes_bev_[j][1], 1);
                    cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), bev_color, 2);
                }

                cv::line(canvas, cv::Point(center_canvas[0], center_canvas[1]), cv::Point(head_canvas[0], head_canvas[1]), bev_color, 2);
            }
        }

        cv::Mat show_img = cv::Mat::zeros(vimgs[0].rows * 2 + canvas_size, vimgs[0].cols * 3, CV_8UC3);

        #pragma omp parallel for num_threads(openmp_num_threads_)
        for(int h = 0; h < 2; ++h) {
            const int idxh = h * 3;

            for(int w = 0; w < 3; ++w) {
                const int idxw = idxh + w;
                vimgs[idxw].copyTo(show_img.rowRange(vimgs[0].rows * h + (canvas_size * h), vimgs[0].rows * (h + 1) + (canvas_size * h)).colRange(vimgs[0].cols * w, vimgs[0].cols * (w + 1)));
            }
        }

        const int w_begin = (vimgs[0].cols * 3 - canvas_size) / 2;
        canvas.copyTo(show_img.rowRange(vimgs[0].rows, vimgs[0].rows + canvas_size).colRange(w_begin, w_begin + canvas_size));
        cv::resize(show_img, show_img, cv::Size(int(vimgs[0].cols / 4 * 3), int(vimgs[0].rows / 4 * 2 + canvas_size / 4)));
        std::filesystem::create_directories(vis_output_dir_);
        std::stringstream vis_name;
        vis_name << "vis_" << std::setw(4) << std::setfill('0') << frame_index_ << ".png";
        cv::imwrite((std::filesystem::path(vis_output_dir_) / vis_name.str()).string(), show_img);
    }
}

void TRTBEVDet::inference(std::vector<cv::Mat> imgs) {
    preprocess(imgs);

    for(int i = 0; i < cuda_in_.size(); ++i) {
        cuda_buff_[i] = cuda_in_[i];
    }

    for(int i = 0; i < cuda_out_.size(); ++i) {
        cuda_buff_[cuda_in_.size() + i] = cuda_out_[i];
    }

    cudaMemcpyAsync(cuda_in_[0], (void*)tensor_in_.get(), size_in_[0], cudaMemcpyHostToDevice, stream_[0]);
    cudaMemcpyAsync(cuda_in_[1], (void*)ranks_depth_.get(), size_in_[1], cudaMemcpyHostToDevice, stream_[1]);
    cudaMemcpyAsync(cuda_in_[2], (void*)ranks_feat_.get(), size_in_[2], cudaMemcpyHostToDevice, stream_[2]);
    cudaMemcpyAsync(cuda_in_[3], (void*)ranks_bev_.get(), size_in_[3], cudaMemcpyHostToDevice, stream_[3]);
    cudaMemcpyAsync(cuda_in_[4], (void*)interval_starts_.get(), size_in_[4], cudaMemcpyHostToDevice, stream_[4]);
    cudaMemcpyAsync(cuda_in_[5], (void*)interval_lengths_.get(), size_in_[5], cudaMemcpyHostToDevice, stream_[5]);

    for(int i = 0; i < stream_.size(); ++i) {
        cudaStreamSynchronize(stream_[i]);
    }

    context_->enqueueV2(cuda_buff_.data(), stream_[0], nullptr);
    cudaStreamSynchronize(stream_[0]);

    // result_serialize layout: per-task [reg, height, dim, rot, vel, heatmap]
    for(int t = 0; t < num_task_; ++t) {
        const int b = t * 6;
        cudaMemcpyAsync(reg_[t].get(),     cuda_out_[b + 0], size_out_[b + 0], cudaMemcpyDeviceToHost, stream_[0]);
        cudaMemcpyAsync(height_[t].get(),  cuda_out_[b + 1], size_out_[b + 1], cudaMemcpyDeviceToHost, stream_[1]);
        cudaMemcpyAsync(dim_[t].get(),     cuda_out_[b + 2], size_out_[b + 2], cudaMemcpyDeviceToHost, stream_[2]);
        cudaMemcpyAsync(rot_[t].get(),     cuda_out_[b + 3], size_out_[b + 3], cudaMemcpyDeviceToHost, stream_[3]);
        cudaMemcpyAsync(vel_[t].get(),     cuda_out_[b + 4], size_out_[b + 4], cudaMemcpyDeviceToHost, stream_[4]);
        cudaMemcpyAsync(heatmap_[t].get(), cuda_out_[b + 5], size_out_[b + 5], cudaMemcpyDeviceToHost, stream_[5]);
    }

    for(int i = 0; i < stream_.size(); ++i) {
        cudaStreamSynchronize(stream_[i]);
    }

    postprocess(imgs);
}