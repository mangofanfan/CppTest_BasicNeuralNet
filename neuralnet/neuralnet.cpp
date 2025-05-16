#include <iostream>
#include <vector>
#include <cmath>
#include "neuralnet.h"

namespace neuralnet {
    NeuralNet::NeuralNet(std::vector<int>& layersNum, double rate, double mobp) : mobp(mobp), rate(rate) {
        numLayers = layersNum.size(); // 输入层和输出层
        layers.resize(layersNum.size());
        layerErr.resize(layersNum.size());
        layer_weight.resize(layersNum.size());
        layer_weight_delta.resize(layersNum.size());
        
        for (size_t l = 0; l < layersNum.size(); ++l) {
            layers[l].resize(layersNum[l]);
            layerErr[l].resize(layersNum[l]);
            if (l + 1 < layersNum.size()) {
                layer_weight[l].resize(layersNum[l] + 1, std::vector<double>(layersNum[l + 1]));
                layer_weight_delta[l].resize(layersNum[l] + 1, std::vector<double>(layersNum[l + 1]));
                for (int j = 0; j <= layersNum[l]; ++j) {
                    for (int i = 0; i < layersNum[l + 1]; ++i) {
                        layer_weight[l][j][i] = randdouble(0.0, 1.0);
                    }
                }
            }
        }
    }

    NeuralNet::~NeuralNet() {
        std::cout << "NeuralNet destructor called. Bye!" << std::endl;
    }

    std::vector<double> NeuralNet::inference(const std::vector<int> &input) {
        // 前向传播
        this->forward(input);
        // 返回输出层的值
        return this->layers.back();
    }

    void NeuralNet::forward(const std::vector<int> &input) {
        for (size_t l = 1; l < layers.size(); ++l) {
            for (int j = 0; j < layers[l].size(); ++j) {
                double z = layer_weight[l-1].back()[j]; // Last row of previous weight matrix
                for (int i = 0; i < layers[l-1].size(); ++i) {
                    z += layer_weight[l-1][i][j] * (l == 1 ? input[i] : layers[l-1][i]);
                }
                layers[l][j] = 1.0 / (1.0 + exp(-z));
            }
        }
    }

    void NeuralNet::backward(const std::vector<int> &target) {
        int l = layers.size() - 1;
        for (int j = 0; j < layerErr[l].size(); ++j) {
            layerErr[l][j] = layers[l][j] * (1 - layers[l][j]) * (target[j] - layers[l][j]);
        }

        while (l-- > 0) {
            for (int j = 0; j < layerErr[l].size(); ++j) {
                double z = 0.0;
                for (size_t i = 0; i < layerErr[l+1].size(); ++i) {
                    z += (l > 0 ? layerErr[l+1][i] : 0) * layer_weight[l][j][i];
                    layer_weight_delta[l][j][i] = mobp * layer_weight_delta[l][j][i] + rate * layerErr[l+1][i] * layers[l][j];
                    layer_weight[l][j][i] += layer_weight_delta[l][j][i];
                    if (j == layerErr[l].size() - 1) {
                        layer_weight_delta[l][j+1][i] = mobp * layer_weight_delta[l][j+1][i] + rate * layerErr[l+1][i];
                        layer_weight[l][j+1][i] += layer_weight_delta[l][j+1][i];
                    }
                }
                layerErr[l][j] = z * layers[l][j] * (1 - layers[l][j]);
            }
        }
    }

    void NeuralNet::train(const std::vector<std::vector<int>>& input, const std::vector<std::vector<int>>& target) {
        for (int i = 0; i < input.size(); ++i) {
            this->forward(input[i]);
            this->backward(target[i]);
        }
    }
}