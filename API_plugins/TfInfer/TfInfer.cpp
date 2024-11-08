#include "TfInfer.h"
#include <unordered_map>

PrePost* createTf(const char* model_path, const std::vector<size_t>& out_sizes) {
    return new TfInfer(model_path,out_sizes);
}

void TfInfer::LoadGraph() {
    tensorflow::SessionOptions options; 
    tensorflow::RunOptions run_opts;
    auto load_status = ReadBinaryProto(tensorflow::Env::Default(), model_path_, &graph_def);
    if (!load_status.ok()) {
        printf("couldn't load the graph\n");
        return;
    }
    session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    auto session_create_status = session->Create(graph_def);
    if (!session_create_status.ok()) {
        printf("couldn't create session\n");
    }
}

TfInfer::TfInfer(const char* model_path, const std::vector<size_t>& out_sizes) : 
                    model_path_{model_path},
                    output_sizes_def{out_sizes}
{
    LoadGraph();
    graph_size = graph_def.node_size();
    for(size_t i =0; i< graph_size; ++i){
        tensorflow::NodeDef node = graph_def.node(i);
        node_map[node.name()] = node;
    }
    record_tensor_details();    
}

std::vector<tensorflow::NodeDef> TfInfer::inbound(tensorflow::NodeDef& node){
    std::vector<tensorflow::NodeDef> ret_vec;
    for(auto name: node.input()){
        auto inp_node = node_map[name];
        ret_vec.push_back(inp_node);
    }
    return ret_vec;
}

void TfInfer::record_tensor_details(){
    num_inputs = 0;
    for(size_t i =0; i< graph_size; ++i){
        tensorflow::NodeDef node = graph_def.node(i);
        if(node.op() == "Placeholder"){
            input_names.push_back(node.name());
            tensorflow::TensorShape shape = node.attr().at("shape").shape();
            std::vector<int64_t> cur_shapes;
            for(int i=0; i<shape.dims(); ++i){
                cur_shapes.push_back(shape.dim_size(i));
            }
            input_shapes.push_back(cur_shapes);
            input_sizes.push_back(static_cast<size_t>(shape.num_elements()));
            num_inputs++;
            tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,shape);
            model_inputs.push_back({node.name(), input_tensor});
        }
    }
    for(size_t i =0; i< graph_size; ++i){
        tensorflow::NodeDef node = graph_def.node(i);
        for(tensorflow::NodeDef in_node: inbound(node)){
            outbound_node_map[in_node.name()].push_back(node);
        }
    }
    std::vector<tensorflow::NodeDef> no_outbound;
    for(size_t i =0; i< graph_size; ++i){
        tensorflow::NodeDef node = graph_def.node(i);
        if(outbound_node_map.find(node.name())==outbound_node_map.end()){
            if(node.op()!="Assert" && node.op()!="NoOp"){
                no_outbound.push_back(node);
            }
        }
    }
    num_outputs = 0;
    for(tensorflow::NodeDef out_node: no_outbound){
        num_outputs++;
        output_names.push_back(out_node.name());
    }
    if(output_sizes_def.size() == num_outputs){
        output_sizes = output_sizes_def;
        return;
    }
    for(tensorflow::NodeDef out_node: no_outbound){
        if(out_node.attr().find("_output_shapes") == out_node.attr().end()){
            std::ostringstream oss;
            oss << "Output shapes of the model, "<<model_path_<<" couldn't be found. Please give the sizes of the outputs in ";
            oss << "connect_pre_model() in the following order: \n";
            for(int i=0; i<num_outputs;++i){
                oss<<output_names[i]<<"\n";
            }
            throw(std::runtime_error(oss.str()));
        }
        tensorflow::TensorShape shape = out_node.attr().at("_output_shapes").list().shape()[0];
        std::vector<int64_t> cur_shapes;
        for(int i=0; i<shape.dims(); ++i){
            if(shape.dim_size(i)<0)
            dynamic_output = true;
            cur_shapes.push_back(shape.dim_size(i));
        }
        output_shapes.push_back(cur_shapes);
        output_sizes.push_back(static_cast<size_t>(shape.num_elements()));
        tensorflow::Tensor output_tensor(tensorflow::DT_FLOAT,shape);
        model_outputs.push_back(output_tensor);
    }
}

void TfInfer::runinference(std::vector<MX::Types::FeatureMap<float>*> inputs, std::vector<MX::Types::FeatureMap<float>*> outputs){
    for(int i =0; i<num_inputs;++i ){
        inputs[i]->get_data((float*)model_inputs[i].second.data());
    }
    
    tensorflow::Status run_status = session->Run(model_inputs, output_names, {}, &model_outputs);

    for(int i =0; i<num_outputs;++i ){
        outputs[i]->set_data((float*)model_outputs[i].data());
    }
}

void TfInfer::runinference(std::vector<MX::Types::FeatureMap<uint8_t>*> input, std::vector<MX::Types::FeatureMap<uint8_t>*> output){}

std::vector<std::vector<int64_t>> TfInfer::get_input_shapes(){
    return input_shapes;
}
std::vector<std::vector<int64_t>> TfInfer::get_output_shapes(){
    return output_shapes;
}
std::vector<size_t>  TfInfer::get_output_sizes(){
    return output_sizes;
}
std::vector<size_t>  TfInfer::get_input_sizes(){
    return input_sizes;
}

std::vector<std::string> TfInfer::get_input_names() {
    return input_names;
}

std::vector<std::string> TfInfer::get_output_names() {
    return output_names;
}

// void TfInfer::record_tensor_details(){
//     const auto& signature_def_map = bundle.GetSignatures();
//     const auto& signature_def = signature_def_map.at("serving_default");
//     num_inputs = signature_def.inputs().size();
//     for (int i = 0; i< num_inputs; ++i) {
//         std::string key;
//         if(num_inputs==1)
//         key="input";
//         else{
//             key="input_";
//             key+=to_string(i);
//         }
//         tensorflow::TensorInfo info = signature_def.inputs().at(key);
//         input_names.push_back(info.name());
//         tensorflow::TensorShape shape = info.tensor_shape();
//         input_shapes_.push_back(shape);
//         std::vector<int64_t> temp_shape;
//         for(int i=0; i<shape.dims();++i){
//             temp_shape.push_back(shape.dim_size(i));
//         }
//         input_shapes.push_back(temp_shape);
//         input_sizes.push_back(static_cast<size_t>(shape.num_elements()));
//     }
//     num_outputs = signature_def.outputs().size();
//     for (int i = 0; i< num_outputs; ++i) {
//         std::string key;
//         if(num_outputs==1)
//         key="output";
//         else{
//             key="output_";
//             key+=to_string(i);
//         }
//         tensorflow::TensorInfo info = signature_def.outputs().at(key);
//         tensorflow::TensorShape shape = info.tensor_shape();
//         output_names.push_back(info.name());
//         output_shapes_.push_back(shape);
//         std::vector<int64_t> temp_shape;
//         for(int i=0; i<shape.dims();++i){
//             temp_shape.push_back(shape.dim_size(i));
//         }
//         output_shapes.push_back(temp_shape);
//         output_sizes.push_back(static_cast<size_t>(shape.num_elements()));
//     }
// }

// void TfInfer::runinference(std::vector<MX::Types::FeatureMap<float>*> inputs, std::vector<MX::Types::FeatureMap<float>*> outputs){
//     std::vector<std::pair<std::string, tensorflow::Tensor> > model_inputs;
//     for(int i =0; i<num_inputs;++i ){
//         tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,input_shapes_[i]);
//         model_inputs.push_back({input_names[i], input_tensor});
//         inputs[i]->get_data((float*)input_tensor.data());
//     }
//     // Populate inputs vector with your input data
//     std::vector<tensorflow::Tensor> model_outputs;
//     for(int i =0; i<num_outputs;++i ){
//         tensorflow::Tensor output_tensor(tensorflow::DT_FLOAT,output_shapes_[i]);
//         model_outputs.push_back(output_tensor);
//         outputs[i]->set_data((float*)output_tensor.data());
//     }

//     tensorflow::Status run_status = bundle.GetSession()->Run(model_inputs, output_names, {}, &model_outputs);

//     for(int i =0; i<num_outputs;++i ){
//         outputs[i]->set_data((float*)model_outputs[i].data());
//     }
// }