# 🔍 Deep Learning Model Profiling

本项目是西电课程作业 **《基于TorchProf的深度学习模型性能分析》** 的完整实现，旨在通过 [TorchProf](https://github.com/awwong1/torchprof) 对多种深度学习模型进行逐层性能剖析，定位性能瓶颈并探索优化策略。

由于该项目的 PyTorch 版本问题，我用 AI 工具重写了一版 [TorchProf_xdu](https://github.com/Livinfly/torchprof_xdu) 可供参考。

> 📄 详细内容与要求请参考[课程提供文档](基于TorchProf的深度学习模型性能分析.pdf)

```bash
# 由于包含引用的repo
git clone --recursive <url_主仓库>
```

## 📚Tasks

运行后，均保存输出完整 log 至 results 文件夹，文本、图片（截图）格式都存一下，为后续报告做准备。

按照**任务_设置参数**进行命名，默认输入也需标明，同时存储 `*_trace.json` 文件。

默认有 `cuda` 用 `cuda` 因为要测量 `cpu`，记得测一版 `--use_cpu`，`cuda` 版本的cpu耗时等是 `cuda` 时的部分耗时。

注意标明是 `cpu` 版本结果还是 `cuda` 能具体标明显卡、cpu是更好的，最后转移到 results 中。

trace 解析方式 chrome://tracing/

```bash
# Task1
## 由于直接使用 torch.profiler，跳过

## 按需
## [--trace_export --use_cpu]

# Task2 
## 使用默认参数即可，可以试着导出 --trace_export
python '.\profile_tasks\task2&3.py' --model ori --trace_export


# Task3
## 1. 定位性能瓶颈
python '.\profile_tasks\task2&3.py' --model ori --sorted --row_limit 10 --trace_export

### 除了完整的分析外，还会根据 cpu, gpu 的时间耗费、内存占用分别排序
### 目前自己实现的排序有些问题，请更具结果，对照去是哪一层，或者看 chrome trace
### 摘取需要处理的性能瓶颈
### 我的版本的 opt 模型以 CPU Mem 最高的层 features_Conv2d_0 为例（综合考虑修改难度等等）

## 2. 优化性能，产生新模型，分析新性能
python '.\profile_tasks\task2&3.py' --model opt --row_limit 10 --trace_export


# Task4
## 1. 测试 ResNet-18 和 MobileNetV2
python .\profile_tasks\task4.py --model resnet18 --trace_export
python .\profile_tasks\task4.py --model mobilenet_v2 --trace_export

## 2. 修改输入形状，若要 trace_export 建议手动修改导出名称 --trace_export
python .\profile_tasks\task4.py --model resnet18 --batch_size 32
python .\profile_tasks\task4.py --model mobilenet_v2 --batch_size 32

## 3. 多输入模型分析
python .\profile_tasks\task4.py --model mim --trace_export

## 4. 不同分辨率，若要 trace_export 建议手动修改导出名称 --trace_export
python .\profile_tasks\task4.py --model resnet18 --H 512 --W 512
python .\profile_tasks\task4.py --model mobilenet_v2 --H 512 --W 512


# Task5
## 1. 选择并准备模型与数据
### 属于准备阶段，在 http://cs231n.stanford.edu/tiny-imagenet-200.zip
### 手动下载 tiny-imagenet-200 数据集，放在 data 文件夹下并解压
### 文件结构 data/tiny-imagenet-200/[train, val, test, ...]

## 按需
## [--upper_simple --trace_export --use_cpu]

## 2. 细粒度的性能剖析，根据要求调整不同参数
python .\profile_tasks\task5.py --model resnet50 --batch_size 1
python .\profile_tasks\task5.py --model mobilenet_v3_large --batch_size 1
python .\profile_tasks\task5.py --model vit_b_16 --batch_size 1

## 3. 自行编写获得画图所需的数据，并做后续处理
### 在 raw_draw 函数中完成画图、列表等工作
### 自行处理，比如提取出 dict 后就自己处理
### 就自己人眼看，最后可视化一下，建议 资源分布饼图/条形图，Top-K 耗时层
```

## 📌 许可证

本项目仅用于课程学习与学术交流，代码基于 MIT 协议开放。请勿用于商业用途。

## ⚠️ 免责说明

该课程作业源码仅供参考，本人不提倡完全抄袭，只是提供思路与实现方式的参考。

请各位同学根据自身理解独立完成课程任务，合理使用本项目内容。

若他人因复制或使用本项目代码而导致课程成绩受影响或其他不良后果，概与本人无关，本人不承担任何责任。
