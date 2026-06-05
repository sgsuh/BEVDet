/*
Create: 2026.05.30
Author: SG.SUH
*/

#pragma once

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include "trt_wrapper.h"

class TRTBEVDet : public TRTWrapper {
public:
    TRTBEVDet(YAML::Node& node);
    virtual ~TRTBEVDet();

    void inference(std::vector<cv::Mat> imgs);
    void allocMat(const char* path, std::unique_ptr<float[]>& data);

    std::unique_ptr<float[]> ego2global_rot_;
    std::unique_ptr<float[]> ego2global_trans_;
    std::unique_ptr<float[]> lidar2ego_rot_;
    std::unique_ptr<float[]> lidar2ego_trans_;

    std::vector<std::unique_ptr<float[]>> sensor2lidar_rot_;
    std::vector<std::unique_ptr<float[]>> sensor2lidar_trans_;
    std::vector<std::unique_ptr<float[]>> cam_intrin_;

    int frame_index_ = 0;
    std::vector<cv::Mat> vis_imgs_;  // optional originals for vis; postprocess uses these when non-empty

protected:
    void allocMeta(const char* path, std::unique_ptr<int[]>& data);
    void preprocess(std::vector<cv::Mat>& imgs);
    void tensorFromImg(std::vector<cv::Mat> imgs);
    void postprocess(std::vector<cv::Mat> imgs);
    void getBboxes(std::vector<std::vector<std::vector<float>>>& boxes_3d,
                    std::vector<std::vector<float>>& scores_3d,
                    std::vector<std::vector<float>>& labels_3d);
    void decode(std::vector<std::pair<int, float>>& heat,
                float* rot,
                float* hei,
                float* dim,
                float* vel,
                float* reg,
                int task_id,
                std::vector<std::vector<float>>& final_bboxes,
                std::vector<float>& final_scores,
                std::vector<float>& final_labels);
    void topk(std::vector<std::pair<int, float>>& heat,
            int k,
            int task_id,
            float* scores,
            float* inds,
            float* clses,
            float* ys,
            float* xs);
    void getTaskDetections(int num_class_with_bg,
                            std::vector<float>& cls_preds,
                            std::vector<std::vector<float>>& reg_preds,
                            std::vector<float>& cls_labels,
                            int task_id);
    void xywhr2xyxyr(std::vector<std::vector<float>>& boxes_xywhr,
                    std::vector<std::vector<float>>& boxes_xyxyr);

    std::vector<int> circleNms(std::vector<std::vector<float>>& final_bboxes,
                                const float thresh,
                                const int post_max_size);

    bool init_;
    bool vis_;
    std::string vis_output_dir_;

    int num_cam_;
    int num_task_;
    int resize_width_;
    int resize_height_;
    int crop_left_;
    int crop_right_;
    int crop_top_;
    int crop_bottom_;
    int post_max_size_;
    int code_size_;
    int max_num_;
    int out_size_factor_;
    int pre_max_size_;
    int openmp_num_threads_;

    float score_threshold_;

    std::unique_ptr<int[]> ranks_bev_;
    std::unique_ptr<int[]> ranks_depth_;
    std::unique_ptr<int[]> ranks_feat_;
    std::unique_ptr<int[]> interval_starts_;
    std::unique_ptr<int[]> interval_lengths_;
    std::unique_ptr<float[]> tensor_in_;

    std::vector<int> num_class_;
    std::vector<int> nms_type_;
    std::vector<float> min_radius_;
    std::vector<float> voxel_size_;
    std::vector<float> pc_range_;
    std::vector<float> post_center_range_;
    std::vector<float> nms_thresh_;
    std::vector<std::vector<float>> nms_rescale_factor_;
    std::vector<std::unique_ptr<float[]>> reg_;
    std::vector<std::unique_ptr<float[]>> height_;
    std::vector<std::unique_ptr<float[]>> dim_;
    std::vector<std::unique_ptr<float[]>> rot_;
    std::vector<std::unique_ptr<float[]>> vel_;
    std::vector<std::unique_ptr<float[]>> heatmap_;
    std::vector<std::vector<int>> draw_boxes_indexes_;
    std::vector<std::vector<int>> draw_boxes_indexes_bev_;
    std::vector<std::vector<int>> color_map_;
    std::vector<float> mean_;
    std::vector<float> std_;
    std::vector<std::vector<float>> corners_norm_;

    std::array<cudaStream_t, 6> stream_;
};