#include <random>
#include <iostream>
#include "neuralnet.h"

double randdouble(double min=-0.0, double max=1.0) {
    // 使用随机数引擎生成随机数
    std::random_device rd;  // 获取随机数种子
    std::mt19937 gen(rd()); // 初始化随机数引擎
    std::uniform_real_distribution<double> dis(min, max); // 定义均匀分布
    double random_number = dis(gen); // 生成随机数
    // std::clog << "Random number generated: " << random_number << std::endl; // 输出生成的随机数
    return random_number; // 返回生成的随机数
}