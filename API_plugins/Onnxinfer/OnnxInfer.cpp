#include "OnnxInfer.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <thread>
#include <chrono>

PrePost* createOnnx(const char* model_path, const std::vector<size_t>& out_sizes) {
    return new OnnxInfer(model_path, out_sizes);
}

void OnnxInfer::init_obj(const OrtApi  g_ort, onnx_struct& onnx_obj,size_t size, Mode mode)
{    
    onnx_obj.node_names.resize(size);
    onnx_obj.node_dims.resize(size);
    onnx_obj.node_types.resize(size);
    onnx_obj.tensor_sizes.resize(size);

    for (size_t i = 0; i < size; i++)
    {
        // Get input node names
        Ort::AllocatorWithDefaultOptions allocator;
        char* obj_name;
        if(mode==Mode::Input){
            g_ort.SessionGetInputName(*session, i, allocator, &obj_name);
        }
        else{
            g_ort.SessionGetOutputName(*session, i, allocator, &obj_name);
        }

        onnx_obj.node_names[i] = obj_name;
        // Get input node types
        OrtTypeInfo* typeinfo;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        if(mode==Mode::Input){
            g_ort.SessionGetInputTypeInfo(*session, i, &typeinfo);
        }
        else{
            g_ort.SessionGetOutputTypeInfo(*session, i, &typeinfo);
        }
        g_ort.CastTypeInfoToTensorInfo(typeinfo, &tensor_info);

        // Get input shapes/dims
        size_t num_dims;
        g_ort.GetDimensionsCount(tensor_info, &num_dims);
        onnx_obj.node_dims[i].resize(num_dims);
        g_ort.GetDimensions(tensor_info, onnx_obj.node_dims[i].data(), num_dims);

        // Get tensor size
        if(onnx_obj.node_dims[i][0]>0){
            size_t tensor_size;
            g_ort.GetTensorShapeElementCount(tensor_info, &tensor_size);
            onnx_obj.tensor_sizes[i] = tensor_size;
        }
        else{
            onnx_obj.tensor_sizes[i] = 0;
        }

        // Release typeinfo
        if (typeinfo) g_ort.ReleaseTypeInfo(typeinfo);
    }
}

OnnxInfer::OnnxInfer(const char* _model_path, const std::vector<size_t>& out_sizes): model_path{_model_path}    
{
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.DisableMemPattern();
    sessionOptions.DisableCpuMemArena();
    sessionOptions.DisableProfiling();
    sessionOptions.DisablePerSessionThreads();
    sessionOptions.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    sessionOptions.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning, "0");
    sessionOptions.AddConfigEntry(kOrtSessionOptionsUseDeviceAllocatorForInitializers,"1");
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL);

    OrtEnv* environment;
    OrtThreadingOptions* envOpts;
    const OrtApi g_ort = Ort::GetApi();
    g_ort.CreateThreadingOptions(&envOpts);
    g_ort.SetGlobalSpinControl(envOpts,0);
    g_ort.SetGlobalInterOpNumThreads(envOpts,1);
    g_ort.SetGlobalIntraOpNumThreads(envOpts,1);
    g_ort.CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_FATAL,"ort_logger",envOpts,&environment);
    g_ort.DisableTelemetryEvents(environment);

    env = new Ort::Env(environment);
#ifndef OS_LINUX
    std::string model_path_(model_path);
    std::wstring widestr = std::wstring(model_path_.begin(), model_path_.end());
    const wchar_t* widecstr = widestr.c_str();
    session = new Ort::Session(*env, widecstr, sessionOptions);
#else
    session = new Ort::Session(*env, model_path, sessionOptions);
#endif
    num_input_nodes = session->GetInputCount();
    num_output_nodes = session->GetOutputCount();


    init_obj(g_ort,input_struct,num_input_nodes,Mode::Input);
    init_obj(g_ort,output_struct,num_output_nodes,Mode::Output);
    if(output_struct.node_dims[0][0]<0){
        dynamic_output = true;
    }

    runOpts.SetRunLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);
    runOpts.SetRunLogVerbosityLevel(ORT_LOGGING_LEVEL_FATAL);
}

void OnnxInfer::runinference(std::vector<MX::Types::FeatureMap<float>*> input, std::vector<MX::Types::FeatureMap<float>*> output){

    for (size_t i = 0; i < num_input_nodes; i++)
    {
        inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
            memoryInfo, input[i]->get_data_ptr(), (input_struct.tensor_sizes[i]), 
            input_struct.node_dims[i].data(), input_struct.node_dims[i].size()));
    }   

    outputTensors = session->Run(runOpts, 
                input_struct.node_names.data(), inputTensors.data(), num_input_nodes, 
                output_struct.node_names.data(), num_output_nodes);
    for(size_t j = 0; j<num_output_nodes; ++j){
        if(dynamic_output){ 
            output_struct.tensor_sizes[j] =  outputTensors[j].GetTensorTypeAndShapeInfo().GetElementCount();
            output_struct.node_dims[j] = outputTensors[j].GetTensorTypeAndShapeInfo().GetShape();
            if(output_struct.tensor_sizes[j]>0)
            memcpy(output[j]->get_data_ptr(),outputTensors[j].GetTensorData<float>(),sizeof(float)*output_struct.tensor_sizes[j]);
        }
        else{
            output[j]->set_data(outputTensors[j].GetTensorMutableData<float>());
        }
    }
}

// void OnnxInfer::runinference(std::vector<MX::Types::FeatureMap<uint8_t>*> input, std::vector<MX::Types::FeatureMap<uint8_t>*> output){

//     for (size_t i = 0; i < num_input_nodes; i++)
//     {
//         inputTensors.emplace_back(Ort::Value::CreateTensor<uint8_t>(
//             memoryInfo, input[i]->get_data_ptr(), (input_struct.tensor_sizes[i]), 
//             input_struct.node_dims[i].data(), input_struct.node_dims[i].size()));
//     }   

//     outputTensors = session->Run(runOpts, 
//                 input_struct.node_names.data(), inputTensors.data(), num_input_nodes, 
//                 output_struct.node_names.data(), num_output_nodes);
//     for(size_t j = 0; j<num_output_nodes; ++j){
//         if(dynamic_output){ 
//             output_struct.tensor_sizes[j] =  outputTensors[j].GetTensorTypeAndShapeInfo().GetElementCount();
//             output_struct.node_dims[j] = outputTensors[j].GetTensorTypeAndShapeInfo().GetShape();
//             if(output_struct.tensor_sizes[j]>0)
//             memcpy(output[j]->get_data_ptr(),outputTensors[j].GetTensorData<float>(),sizeof(float)*output_struct.tensor_sizes[j]);
//         }
//     }
// }

std::vector<std::vector<int64_t>> OnnxInfer::get_output_shapes(){
    return output_struct.node_dims;
}

std::vector<std::vector<int64_t>> OnnxInfer::get_input_shapes(){
    return input_struct.node_dims;
}

std::vector<size_t> OnnxInfer::get_output_sizes(){
    return output_struct.tensor_sizes;
}

std::vector<size_t> OnnxInfer::get_input_sizes(){
    return input_struct.tensor_sizes;
}

std::vector<std::string> OnnxInfer::get_input_names(){
    std::vector<std::string> in_names;
    for(int i = 0; i<num_input_nodes; ++i)
    in_names.push_back(input_struct.node_names[i]);
    return in_names;
}

std::vector<std::string> OnnxInfer::get_output_names(){
    std::vector<std::string> out_names;
    for(int i = 0; i<num_output_nodes; ++i)
    out_names.push_back(output_struct.node_names[i]);
    return out_names;
}

OnnxInfer::~OnnxInfer(){
    for(int i = 0; i<num_input_nodes; ++i)
    free(input_struct.node_names[i]);
    for(int i = 0; i<num_output_nodes; ++i)
    free(output_struct.node_names[i]);
    delete session;
}