# C++ dynamics parameter sampler

这个目录提供 `scripts/sample_dynamics_parameters.py` 的 C++17 版本，只负责随机生成四旋翼动力学参数 JSON 文件。

编译：

```bash
g++ -std=c++17 -O2 cpp/sample_dynamics_parameters.cpp -o sample_dynamics_parameters_cpp
```

运行：

```bash
./sample_dynamics_parameters_cpp --num 10 --seed 0 --output-dir test_output_cpp
```

输出文件形如：

```text
test_output_cpp/0.json
test_output_cpp/1.json
test_output_cpp/2.json
```

程序不会训练 teacher policy，不会做 student policy 蒸馏，也不会生成 RAPTOR foundation policy 的 2084 组最终参数。
