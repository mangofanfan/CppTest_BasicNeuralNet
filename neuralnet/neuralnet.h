#include <iostream>
#include <vector>

namespace neuralnet {
    // 神经网络！
    class NeuralNet {
    private:
        // 神经网络的层数
        int numLayers;

        double mobp; // 动量因子
        double rate; // 学习率

        // 记录节点值（输入层与隐藏层）
        std::vector<std::vector<double>> layers;

        // 记录节点误差
        std::vector<std::vector<double>> layerErr;

        // 记录节点权重
        std::vector<std::vector<std::vector<double>>> layer_weight;

        // 记录节点偏置
        std::vector<std::vector<std::vector<double>>> layer_weight_delta;

        // 前向传播
        void forward(const std::vector<int> &input);

        // 反向传播并修改权重
        void backward(const std::vector<int> &target);
    
    public:
        // 构造函数
        // layers: 各层节点数
        // rate: 学习率
        // mobp: 动量因子
        NeuralNet(std::vector<int>& layers, double rate, double mobp);

        // 析构函数
        ~NeuralNet();

        // 训练！
        void train(const std::vector<std::vector<int>> &input, const std::vector<std::vector<int>> &output);

        // 推理！
        std::vector<double> inference(const std::vector<int> &input);
    };
}

// 随机生成一个范围在[min, max]之间的double类型随机数
double randdouble(double min, double max);