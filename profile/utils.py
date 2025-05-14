import torch
import torch.nn as nn
from torch.profiler import profile
import sys
import os

current_file_path = os.path.abspath(__file__)
profile_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(profile_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torchprof_xdu import ProfileDetailed


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