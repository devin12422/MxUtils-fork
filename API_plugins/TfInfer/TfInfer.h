#ifndef Tf_INFER
#define Tf_INFER

#include <string.h>
#include <memx/accl/prepost.h>
#include <tensorflow/core/public/session.h>

class TfInfer : public PrePost{
    private:
        const char* model_path_;
        std::unique_ptr<tensorflow::Session> session;
        tensorflow::GraphDef graph_def;
        size_t graph_size;
        void record_tensor_details();
        void LoadGraph();
        int num_inputs;
        int num_outputs;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<int64_t>> output_shapes;
        std::vector<size_t> input_sizes;
        std::vector<size_t> output_sizes;
        std::vector<size_t> output_sizes_def;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<tensorflow::TensorShape> input_shapes_;
        std::vector<tensorflow::TensorShape> output_shapes_;
        std::vector<tensorflow::NodeDef> inbound(tensorflow::NodeDef& node);
        std::unordered_map<std::string, tensorflow::NodeDef> node_map;
        std::unordered_map<std::string, std::vector<tensorflow::NodeDef>> outbound_node_map;
        std::vector<std::pair<std::string, tensorflow::Tensor> > model_inputs;
        std::vector<tensorflow::Tensor> model_outputs;
    public:
        ~TfInfer(){};
        TfInfer(const char* model_path, const std::vector<size_t>& out_sizes);
        void runinference(std::vector<MX::Types::FeatureMap<float>*> input, std::vector<MX::Types::FeatureMap<float>*> output) override ;
        std::vector<std::vector<int64_t>> get_input_shapes() override;
        void runinference(std::vector<MX::Types::FeatureMap<uint8_t>*> input, std::vector<MX::Types::FeatureMap<uint8_t>*> output) override;
        std::vector<std::vector<int64_t>> get_output_shapes() override;
        std::vector<size_t> get_output_sizes() override;
        std::vector<size_t> get_input_sizes() override;
        std::vector<std::string> get_output_names() override;
        std::vector<std::string> get_input_names() override;
};

extern "C" {
    PrePost* createTf(const char* model_path, const std::vector<size_t>& out_sizes);
}

#endif
