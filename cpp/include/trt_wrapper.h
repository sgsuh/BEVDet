/*
Create: 2026.05.30
Author: SG.SUH
*/

#pragma once

#include <NvInfer.h>
#include <vector>
#include <yaml-cpp/yaml.h>

class TRTWrapper {
public:
    TRTWrapper();
    virtual ~TRTWrapper();

    void setConfig(const char* cfg_file);

    bool buildEngine();
    bool loadEngine();

    nvinfer1::IExecutionContext* context_;

    std::vector<void*> cuda_in_;
    std::vector<void*> cuda_out_;
    std::vector<void*> cuda_buff_;

    std::vector<nvinfer1::Dims> dim_in_;
    std::vector<nvinfer1::Dims> dim_out_;

    std::vector<int> size_in_;
    std::vector<int> size_out_;

    int batch_size_;

protected:
    void releaseResource();
    void allocResource();

    nvinfer1::ICudaEngine* engine_;

    YAML::Node config_;
};