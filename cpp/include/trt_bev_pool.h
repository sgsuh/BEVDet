/*
Create: 2026.05.30
Author: SG.SUH
*/
#pragma once

#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>
#include "trt_plugin_base.hpp"

class TRTBEVPoolV2 : public TRTPluginBase {
public:
    TRTBEVPoolV2(const std::string& name, int out_width, int out_height);
    TRTBEVPoolV2(const std::string name, const void* data, size_t length);
    TRTBEVPoolV2() = delete;

    ~TRTBEVPoolV2() TRT_NOEXCEPT override = default;

    nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;
    nvinfer1::DimsExprs getOutputDimensions(int output_index,
                                            const nvinfer1::DimsExprs* inputs,
                                            int nb_inputs,
                                            nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;
    
    bool supportsFormatCombination(int pos, 
                                    const nvinfer1::PluginTensorDesc* io_desc, 
                                    int nb_inputs, 
                                    int nb_outputs) TRT_NOEXCEPT override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, 
                        int nb_inputs, 
                        const nvinfer1::DynamicPluginTensorDesc* out, 
                        int nb_outputs) TRT_NOEXCEPT override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, 
                            int nb_inputs, 
                            const nvinfer1::PluginTensorDesc* outputs, 
                            int nb_outputs) const TRT_NOEXCEPT override;

    int enqueue(const nvinfer1::PluginTensorDesc* input_desc, 
                const nvinfer1::PluginTensorDesc* output_desc, 
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) TRT_NOEXCEPT override;

    nvinfer1::DataType getOutputDataType(int index, 
                                        const nvinfer1::DataType* input_types,
                                        int nb_inputs) const TRT_NOEXCEPT override;

    const char* getPluginType() const TRT_NOEXCEPT override;
    const char* getPluginVersion() const TRT_NOEXCEPT override;

    int getNbOutputs() const TRT_NOEXCEPT override;

    size_t getSerializationSize() const TRT_NOEXCEPT override;

    void serialize(void* buffer) const TRT_NOEXCEPT override;

private:
    int out_width_;
    int out_height_;
};

class TRTBEVPoolV2Creator : public TRTPluginCreatorBase {
public:
    TRTBEVPoolV2Creator();

    ~TRTBEVPoolV2Creator() TRT_NOEXCEPT override = default;

    const char* getPluginName() const TRT_NOEXCEPT override;
    const char* getPluginVersion() const TRT_NOEXCEPT override;

    nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                            const void* serialData,
                                            size_t serialLength) TRT_NOEXCEPT override;
};