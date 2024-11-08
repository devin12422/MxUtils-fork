#include "TfliteInfer.h"
#include <tensorflow/lite/logger.h>

PrePost* createTflite(const char* model_path, const std::vector<size_t>& out_sizes) {
    return new TfliteInfer(model_path,out_sizes);
}

TfliteInfer::TfliteInfer(const char* model_path, const std::vector<size_t>& out_sizes): model_path_{model_path}
{
    model = tflite::FlatBufferModel::BuildFromFile(model_path_);

    tflite::LoggerOptions::SetMinimumLogSeverity(tflite::TFLITE_LOG_ERROR);

    if (!model) {
        std::cerr << "Failed to load TFLite model: " << model_path_ << std::endl;
    }

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    record_tensor_details();
    for(int i =0; i< num_outputs; ++i){
        if(output_shapes[i][0]<0){
            dynamic_output = true;
        }
    }
    interpreter->AllocateTensors();
    interpreter->SetNumThreads(0);
}

void TfliteInfer::runinference(std::vector<MX::Types::FeatureMap<float>*> input, std::vector<MX::Types::FeatureMap<float>*> output){
    for(int i=0; i<num_inputs; ++i){
        float* input_tensor = interpreter->typed_input_tensor<float>(i);
        input[i]->get_data(input_tensor);
    }
    interpreter->Invoke();
    for(int i=0; i<num_outputs; ++i){
        float* output_tensor = interpreter->typed_output_tensor<float>(i);
        output[i]->set_data(output_tensor);
    }
}

void TfliteInfer::runinference(std::vector<MX::Types::FeatureMap<uint8_t>*> input, std::vector<MX::Types::FeatureMap<uint8_t>*> output){}

void TfliteInfer::record_tensor_details(){
    num_inputs = interpreter->inputs().size();
    for(int i=0;i<num_inputs;++i){
        TfLiteIntArray* input_dims = interpreter->tensor(interpreter->inputs()[i])->dims;
        vector<int64_t> tmp;
        for(int j =0; j<input_dims->size; ++j){
            tmp.push_back(input_dims->data[j]);
        }
        input_shapes.push_back(tmp);
        input_sizes.push_back(interpreter->tensor(interpreter->inputs()[i])->bytes/ sizeof(float));
        input_names.push_back(interpreter->tensor(interpreter->inputs()[i])->name);
    }

    num_outputs = interpreter->outputs().size();
    for(int i=0;i<num_outputs;++i){
        TfLiteIntArray* output_dims = interpreter->tensor(interpreter->outputs()[i])->dims;
        vector<int64_t> tmp;
        for(int j =0; j<output_dims->size; ++j){
            tmp.push_back(output_dims->data[j]);
        }
        output_shapes.push_back(tmp);
        output_sizes.push_back(interpreter->tensor(interpreter->outputs()[i])->bytes/ sizeof(float));
        output_names.push_back(interpreter->tensor(interpreter->outputs()[i])->name);
    }
}

std::vector<std::vector<int64_t>> TfliteInfer::get_input_shapes(){
    return input_shapes;
}
std::vector<std::vector<int64_t>> TfliteInfer::get_output_shapes(){
    return output_shapes;
}
std::vector<size_t>  TfliteInfer::get_output_sizes(){
    return output_sizes;
}
std::vector<size_t>  TfliteInfer::get_input_sizes(){
    return input_sizes;
}

std::vector<std::string>  TfliteInfer::get_output_names(){
    return output_names;
}
std::vector<std::string>  TfliteInfer::get_input_names(){
    return input_names;
}

TfliteInfer::~TfliteInfer(){
    interpreter.reset();
}