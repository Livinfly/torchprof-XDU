# 🔍 Deep Learning Model Profiling

本项目是西电课程作业 **《基于TorchProf的深度学习模型性能分析》** 的完整实现，旨在通过 [TorchProf](https://github.com/awwong1/torchprof) 对多种深度学习模型进行逐层性能剖析，定位性能瓶颈并探索优化策略。

由于该项目的 PyTorch 版本问题，我用 AI 工具重写了一版 [TorchProf_xdu](https://github.com/Livinfly/torchprof_xdu) 可供参考。

> 📄 详细内容与要求请参考[课程提供文档](基于TorchProf的深度学习模型性能分析.pdf)

```bash
# 由于包含引用的repo
git clone --recursive <url_主仓库>
```

## Tasks

运行后，均保存输出完整 log 至 results 文件夹，文本、图片格式都存一下，为后续报告做准备，按照**任务_设置参数**进行命名，同时存储 `*_trace.json` 文件。

```bash
# Task1
## 由于直接使用 torch.profiler，跳过

# Task2 
## 使用默认参数即可，可以试着导出 --trace_export
python '.\profile\task2&3\main.py' --model ori --trace_export

# Task3
## 1. 定位性能瓶颈
python '.\profile\task2&3\main.py' --model ori --sorted --row_limit 10 --trace_export
### 除了完整的分析外，还会根据 cpu, gpu 的时间耗费、内存占用分别排序
### 目前自己实现的排序有些问题，请更具结果，对照去是哪一层，或者看 chrome trace
### 摘取需要处理的性能瓶颈，我的版本的 opt 模型以 CPU Mem 最高的层 features_Conv2d_0 为例（综合考虑修改难度等等）
## 2. 优化性能，产生新模型，分析新性能
python '.\profile\task2&3\main.py' --model opt --row_limit 10 --trace_export

```

## 📌 许可证

本项目仅用于课程学习与学术交流，代码基于 MIT 协议开放。请勿用于商业用途。

## ⚠️ 免责说明

该课程作业源码仅供参考，本人不提倡完全抄袭，只是提供思路与实现方式的参考。

请各位同学根据自身理解独立完成课程任务，合理使用本项目内容。

若他人因复制或使用本项目代码而导致课程成绩受影响或其他不良后果，概与本人无关，本人不承担任何责任。
