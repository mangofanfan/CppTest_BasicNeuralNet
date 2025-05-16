#include <iostream>
#include <vector>
#include <cmath>
#include "neuralnet.h"
using namespace std;

int main(int, char**){
    // n作为训练次数进行调试
    int m;
    cin >> m;

    // 创建一个神经网络实例
    vector<int> layers = {2, 10, 10, 2}; // 隐藏层节点数
    neuralnet::NeuralNet nn(layers, 0.15, 0.8); // 输入层2个节点，输出层2个节点

    vector<vector<int>> input = {
        {1, 2},
        {2, 2},
        {1, 1},
        {2, 1}
    };

    vector<vector<int>> output = {
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 0}
    };

    for (int n = 0; n < m; ++n) {
        nn.train(input, output);
        if (n % 1000 == 0) {
            cout << "Training iteration: " << n << endl;
        }
    }

    // 获取输出
    input.push_back({3, 0});
    for (int i = 0; i < input.size(); ++i) {
        const auto result = nn.inference(input[i]);
        cout << "Result: (" << input[i][0] << ", " << input[i][1] << ") -> ";
        for (const auto& val : result) {
            if (isnan(val)) {
                cerr << "Error: Output contains nan!" << endl;
            }
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
