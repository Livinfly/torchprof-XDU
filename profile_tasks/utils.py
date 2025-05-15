import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader
import sys
import os

current_file_path = os.path.abspath(__file__)
profile_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(profile_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torchprof_xdu import *

class InputBranch(nn.Module):
    """一个简单的输入分支"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class MultiInputNet(nn.Module):
    """一个接受两个输入的自定义网络"""
    def __init__(self,
                 input_dim_image, hidden_dim_image, branch_output_image,
                 input_dim_text, hidden_dim_text, branch_output_text,
                 shared_hidden_dim, num_classes):
        super().__init__()
        self.image_branch = InputBranch(input_dim_image, hidden_dim_image, branch_output_image)
        self.text_branch = InputBranch(input_dim_text, hidden_dim_text, branch_output_text)
        combined_dim = branch_output_image + branch_output_text
        self.merger_fc1 = nn.Linear(combined_dim, shared_hidden_dim)
        self.merger_activation = nn.ReLU()
        self.classifier = nn.Linear(shared_hidden_dim, num_classes)

    def forward(self, image_input, text_input):
        image_features = self.image_branch(image_input)
        text_features = self.text_branch(text_input)
        combined_features = torch.cat((image_features, text_features), dim=1)
        x = self.merger_fc1(combined_features)
        x = self.merger_activation(x)
        x = self.classifier(x)
        return x


def profile_model(model, model_name, inputs, device, 
                  profiler_activities, row_limit=10, record_shapes=False, 
                  profile_memory=True, with_stack=True, with_flops=True, sorted=False,
                  trace_export=False):
    model.to(device)
    if isinstance(inputs, torch.Tensor):
        inputs_on_device = inputs.to(device)
    elif isinstance(inputs, (list, tuple)):
        inputs_on_device = tuple(inp.to(device) for inp in inputs)
    else:
        raise TypeError(f"Unsupported input type for model: {type(inputs)}")
    model.eval()

    print(f"\n--- Profiling {model_name} on {device} ---")

    print("Warm-up runs (2 iterations)...")
    with torch.no_grad():
        for _ in range(2):
            if isinstance(inputs_on_device, tuple):
                _ = model(*inputs_on_device)
            else:
                _ = model(inputs_on_device)

    print("Starting profiled run...")
    with ProfileDetailed(
        model,
        enabled=True,
        use_cuda=(device.type == "cuda"),
        profile_memory=profile_memory
    ) as prof:
        with torch.no_grad():
            # 为了测试的稳定性，多跑几次？如30次？
            for _ in range(1):
                if isinstance(inputs_on_device, tuple):
                    _ = model(*inputs_on_device)
                else:
                    _ = model(inputs_on_device)

    print(f"\n--- {model_name} Profiler Results ---")
    print(prof.display(top_k=-1))

    if sorted:
        print(f"\n--- {model_name} Profiler Results (Sorted by CPU total time) ---")
        print(prof.display(sort_by='CPU total', top_k=row_limit))

        if device.type == 'cuda':
            print(f"\n--- {model_name} Profiler Results (Sorted by CUDA total time) ---")
            print(prof.display(sort_by="CUDA total", top_k=row_limit))

        print(f"\n--- {model_name} Profiler Results (Sorted by CPU Memory Usage) ---")
        print(prof.display(sort_by="CPU Mem", top_k=row_limit))

        if device.type == 'cuda':
            print(f"\n--- {model_name} Profiler Results (Sorted by CUDA Memory Usage) ---")
            print(prof.display(sort_by="CUDA Mem", top_k=row_limit))
    
    # # (可选) 导出 trace 文件，用于 Chrome Tracing (chrome://tracing)
    # # 目前还是临时通过 torch.profiler 本来就存在的方法导出，重新 proile 一次
    # # 目前 torchprof_xdu 版本不支持直接导出 trace 文件
    # # 为了压低内存使用，trace_export 在主函数运行
    # if trace_export:
    #     trace_file = f"{model_name.replace(' ', '_').lower()}_trace.json"
    #     for _ in range(2):
    #         _ = model(input_tensor)

    #     with profile(
    #         activities=profiler_activities,
    #         record_shapes=record_shapes,
    #         profile_memory=profile_memory,
    #         with_stack=with_stack,
    #         with_flops=with_flops,
    #     ) as prof:
    #         with torch.no_grad():
    #             _ = model(input_tensor)
    #     prof.export_chrome_trace(trace_file)
    #     print(f"\nTrace file exported to: {trace_file}")

def trace_export_func(trace_file, model, sample_input, device, profiler_activities, record_shapes):
    model.to(device)
    if isinstance(sample_input, torch.Tensor):
        inputs_on_device = sample_input.to(device)
    elif isinstance(sample_input, (list, tuple)):
        inputs_on_device = tuple(inp.to(device) for inp in sample_input)
    else:
        raise TypeError(f"Unsupported input type for model: {type(sample_input)}")
    
    with torch.no_grad():
        for _ in range(2):
            if isinstance(inputs_on_device, tuple):
                _ = model(*inputs_on_device)
            else:
                _ = model(inputs_on_device)
    
    with profile(
        activities=profiler_activities,
        record_shapes=record_shapes,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            # 为了测试的稳定性，多跑几次？如30次？
            for _ in range(1):
                if isinstance(inputs_on_device, tuple):
                    _ = model(*inputs_on_device)
                else:
                    _ = model(inputs_on_device)
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace file exported to: {trace_file}")

def trace_export_func_task5(
    trace_file: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    profiler_activities=None,
    record_shapes: bool = False,
    num_warmup_batches: int = 2, # 默认预热批次数为 2
    num_profile_batches: int = 1, # 默认分析批次数为 1
    profile_memory: bool = True,
    with_stack: bool = True,
    with_flops: bool = True
):
    """
    使用 DataLoader 提供的数据对模型进行性能分析，并导出 Chrome 追踪文件。
    输入处理假定 DataLoader 产生的每个批次数据 batch_data 的第一个元素是模型所需的单个输入张量，
    即 inputs, _ = batch_data。

    参数:
        trace_file (str): 保存导出的 Chrome 追踪文件的路径。
        model (torch.nn.Module): 需要进行性能分析的模型。
        dataloader (torch.utils.data.DataLoader): 提供输入数据的 DataLoader。
        device (torch.device): 模型和数据在其上运行的设备。
        profiler_activities (list, optional): 需要追踪的 torch.profiler.ProfilerActivity 列表
            (例如 CPU, CUDA)。如果 CUDA 可用，默认为 [CPU, CUDA]，否则为 [CPU]。
        record_shapes (bool, optional): 是否记录操作符输入的形状。默认为 False。
        num_warmup_batches (int, optional): 在性能分析前用于预热的批次数。默认为 2。
        num_profile_batches (int, optional): 进行性能分析的批次数。默认为 1。
        profile_memory (bool, optional): 是否启用内存性能分析。默认为 True。
        with_stack (bool, optional): 是否记录操作符的调用栈。默认为 True。
        with_flops (bool, optional): 是否估算操作符的浮点运算次数。默认为 True。
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"期望 model 是 torch.nn.Module 的实例，但得到的是 {type(model)}")
    if not isinstance(dataloader, DataLoader):
        raise TypeError(f"期望 dataloader 是 torch.utils.data.DataLoader 的实例，但得到的是 {type(dataloader)}")
    if not isinstance(device, torch.device):
        raise TypeError(f"期望 device 是 torch.device 的实例，但得到的是 {type(device)}")

    model.to(device)
    model.eval() # 设置为评估模式

    if profiler_activities is None:
        profiler_activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            profiler_activities.append(ProfilerActivity.CUDA)

    # --- 预热阶段 ---
    print(f"开始预热，使用 {num_warmup_batches} 个批次...")
    warmup_actual_batches = 0
    if num_warmup_batches > 0:
        with torch.no_grad(): # 预热时不需要计算梯度
            warmup_dataloader_iter = iter(dataloader)
            for i_w in range(num_warmup_batches):
                try:
                    batch_data = next(warmup_dataloader_iter)

                    # 简化输入处理：直接假定 batch_data[0] 是输入张量
                    try:
                        inputs_tensor, _ = batch_data # 假定批次数据是 (输入张量, 其他) 的结构
                    except ValueError:
                        print(f"警告: 预热批次 {i_w} 的数据无法解包为 (输入, _)。请确保 DataLoader产生此格式的数据。跳过此批次。")
                        continue
                        
                    if not isinstance(inputs_tensor, torch.Tensor):
                        print(f"警告: 预热批次 {i_w} 中提取的输入期望是单个 Tensor，但得到的是 {type(inputs_tensor)}。跳过此批次。")
                        continue

                    inputs_on_device = inputs_tensor.to(device, non_blocking=True)
                    _ = model(inputs_on_device) # 模型接收单个张量输入
                    warmup_actual_batches += 1
                except StopIteration:
                    print(f"警告: DataLoader 在预热 {warmup_actual_batches} 个批次后耗尽 (请求 {num_warmup_batches} 个批次)。")
                    break
                except Exception as e:
                    print(f"处理预热批次 {i_w} 时发生错误: {e}")
                    continue

    if warmup_actual_batches < num_warmup_batches and num_warmup_batches > 0:
        print(f"警告: 仅完成了 {warmup_actual_batches} 个预热批次 (请求 {num_warmup_batches} 个)。")
    elif num_warmup_batches > 0:
        print(f"预热完成，共 {warmup_actual_batches} 个批次。")
    else:
        print("num_warmup_batches 为 0，跳过预热阶段。")

    # --- 性能分析阶段 ---
    if num_profile_batches <= 0:
        print("num_profile_batches 非正数，跳过性能分析阶段。")
        return

    print(f"开始性能分析，分析 {num_profile_batches} 个批次...")
    profiled_actual_batches = 0
    profile_dataloader_iter = iter(dataloader)

    with profile(
        activities=profiler_activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    ) as prof:
        with torch.no_grad():
            for i_p in range(num_profile_batches):
                try:
                    with record_function(f"profile_batch_{i_p}"):
                        batch_data = next(profile_dataloader_iter)
                        
                        try:
                            inputs_tensor, _ = batch_data # 假定批次数据是 (输入张量, 其他) 的结构
                        except ValueError:
                            print(f"警告: 分析批次 {i_p} 的数据无法解包为 (输入, _)。请确保 DataLoader产生此格式的数据。跳过此批次。")
                            continue
                        
                        if not isinstance(inputs_tensor, torch.Tensor):
                            print(f"警告: 分析批次 {i_p} 中提取的输入期望是单个 Tensor，但得到的是 {type(inputs_tensor)}。跳过此批次。")
                            continue
                        
                        inputs_on_device = inputs_tensor.to(device, non_blocking=True)
                        _ = model(inputs_on_device) # 模型接收单个张量输入
                        profiled_actual_batches += 1
                except StopIteration:
                    print(f"警告: DataLoader 在分析 {profiled_actual_batches} 个批次后耗尽 (请求 {num_profile_batches} 个批次)。")
                    break
                except Exception as e:
                    print(f"处理分析批次 {i_p} 时发生错误: {e}")
                    continue

    if profiled_actual_batches < num_profile_batches and profiled_actual_batches > 0 :
        print(f"警告: 仅完成了 {profiled_actual_batches} 个性能分析批次 (请求 {num_profile_batches} 个)。")
    elif profiled_actual_batches == 0 and num_profile_batches > 0:
        print(f"错误: 没有批次被成功分析。请检查 dataloader、输入处理逻辑以及 num_profile_batches 设置。")

    if profiled_actual_batches > 0:
        try:
            prof.export_chrome_trace(trace_file)
            print(f"\n追踪文件已导出至: {trace_file}")
        except Exception as e:
            print(f"导出 Chrome 追踪文件时发生错误: {e}")
    else:
        print("\n由于没有批次被成功分析，未导出追踪文件。")