/*
Create: 2026.05.30
Author: SG.SUH
*/

#include "trt_bev_pool.h"
#include <chrono>
#include "trt_bev_pool_kernel.h"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"

namespace {
static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"bev_pool_v2"};
}

TRTBEVPoolV2::TRTBEVPoolV2(const std::string& name, int out_width, int out_height) 
    : TRTPluginBase(name) 
    , out_width_(out_width)
    , out_height_(out_height) {

}

TRTBEVPoolV2::TRTBEVPoolV2(const std::string name, const void* data, size_t length) 
    : TRTPluginBase(name) {
    deserializeValue(&data, &length, &out_width_);
    deserializeValue(&data, &length, &out_height_);
}

nvinfer1::IPluginV2DynamicExt* TRTBEVPoolV2::clone() const TRT_NOEXCEPT {
    TRTBEVPoolV2* plugin = new TRTBEVPoolV2(layer_name_, out_width_, out_height_);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs TRTBEVPoolV2::getOutputDimensions(int output_index,
                                                    const nvinfer1::DimsExprs* inputs,
                                                    int nb_inputs,
                                                    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = expr_builder.constant(1);
    ret.d[1] = expr_builder.constant(out_height_);
    ret.d[2] = expr_builder.constant(out_width_);
    ret.d[3] = inputs[1].d[3];

    return ret;
}

bool TRTBEVPoolV2::supportsFormatCombination(int pos,
                                            const nvinfer1::PluginTensorDesc* io_desc,
                                            int nb_inputs,
                                            int nb_outputs) TRT_NOEXCEPT {
    if(pos == 0 || pos == 1 || pos == 7) {
        return (io_desc[pos].type == nvinfer1::DataType::kFLOAT && io_desc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    } else {
        return (io_desc[pos].type == nvinfer1::DataType::kINT32 && io_desc[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
}

void TRTBEVPoolV2::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                    int nb_inputs,
                                    const nvinfer1::DynamicPluginTensorDesc* outputs,
                                    int nb_outputs) TRT_NOEXCEPT {
    ASSERT(nb_inputs == 7);
    ASSERT(nb_outputs == 1);
}

size_t TRTBEVPoolV2::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                    int nb_inputs,
                                    const nvinfer1::PluginTensorDesc* outputs,
                                    int nb_outputs) const TRT_NOEXCEPT {
    return 0;
}

int TRTBEVPoolV2::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                        const nvinfer1::PluginTensorDesc* output_desc,
                        const void* const* inputs,
                        void* const* outputs,
                        void* work_space,
                        cudaStream_t stream) TRT_NOEXCEPT {
    nvinfer1::Dims feat_dims = input_desc[1].dims;
    nvinfer1::Dims interval_dims = input_desc[5].dims;
    nvinfer1::Dims out_dims = output_desc[0].dims;
    auto data_type = input_desc[0].type;
    int num_points = out_dims.d[0] * out_dims.d[1] * out_dims.d[2] * out_dims.d[3];

    switch(data_type) {
        case nvinfer1::DataType::kFLOAT:
            bevPoolV2SetZero(num_points, (float*)outputs[0]);
            bevPoolV2(feat_dims.d[3], interval_dims.d[0], (float*)inputs[0], (float*)inputs[1], (int*)inputs[2], (int*)inputs[3], (int*)inputs[4], (int*)inputs[5], (int*)inputs[6], (float*)outputs[0], stream);

            break;
        default:
            return 1;
    }

    return 0;
}

nvinfer1::DataType TRTBEVPoolV2::getOutputDataType(int index,
                                                    const nvinfer1::DataType* input_types,
                                                    int nb_inputs) const TRT_NOEXCEPT {
    return input_types[0];
}

const char* TRTBEVPoolV2::getPluginType() const TRT_NOEXCEPT {
    return PLUGIN_NAME;
}

const char* TRTBEVPoolV2::getPluginVersion() const TRT_NOEXCEPT {
    return PLUGIN_VERSION;
}

int TRTBEVPoolV2::getNbOutputs() const TRT_NOEXCEPT {
    return 1;
}

size_t TRTBEVPoolV2::getSerializationSize() const TRT_NOEXCEPT {
    return serializedSize(out_width_) + serializedSize(out_height_);
}

void TRTBEVPoolV2::serialize(void* buffer) const TRT_NOEXCEPT {
    serializeValue(&buffer, out_width_);
    serializeValue(&buffer, out_height_);
}

TRTBEVPoolV2Creator::TRTBEVPoolV2Creator() {
    plugin_attributes_ = std::vector<nvinfer1::PluginField>({nvinfer1::PluginField("output_z"), nvinfer1::PluginField("output_height"), nvinfer1::PluginField("output_width")});
    fc_.nbFields = plugin_attributes_.size();
    fc_.fields = plugin_attributes_.data();
}

const char* TRTBEVPoolV2Creator::getPluginName() const TRT_NOEXCEPT {
    return PLUGIN_NAME;
}

const char* TRTBEVPoolV2Creator::getPluginVersion() const TRT_NOEXCEPT {
    return PLUGIN_VERSION;
}

nvinfer1::IPluginV2* TRTBEVPoolV2Creator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
    int out_width = 128;
    int out_height = 128;

    for(int i = 0; i < fc->nbFields; ++i) {
        if(fc->fields[i].data == nullptr) {
            continue;
        }

        std::string field_name(fc->fields[i].name);

        if(field_name.compare("output_height") == 0) {
            out_height = static_cast<const int*>(fc->fields[i].data)[0];
        }

        if(field_name.compare("output_width") == 0) {
            out_width = static_cast<const int*>(fc->fields[i].data)[0];
        }
    }

    ASSERT(out_height > 0);
    ASSERT(out_width > 0);

    TRTBEVPoolV2* plugin = new TRTBEVPoolV2(name, out_width, out_height);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::IPluginV2* TRTBEVPoolV2Creator::deserializePlugin(const char* name,
                                                            const void* serial_data,
                                                            size_t serial_length) TRT_NOEXCEPT {
    auto plugin = new TRTBEVPoolV2(name, serial_data, serial_length);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTBEVPoolV2Creator);