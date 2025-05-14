import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
# from ...torchprof_xdu.torchprof_xdu_profile_detailed import ProfileDetailed
import argparse
import copy

import sys
import os

current_file_path = os.path.abspath(__file__)
task_dir = os.path.dirname(current_file_path)
profile_dir = os.path.dirname(task_dir)
project_root = os.path.dirname(profile_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from torchprof_xdu import ProfileDetailed

def profile_model(model, model_name, input_tensor, device, 
                  profiler_activities, row_limit=10, record_shapes=False, 
                  profile_memory=True, with_stack=True, with_flops=True, sorted=False,
                  trace_export=False):
    model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    print(f"\n--- Profiling {model_name} on {device} ---")

    print("Warm-up runs (2 iterations)...")
    for _ in range(2):
        _ = model(input_tensor)

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
                _ = model(input_tensor)

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


def create_optimized_alexnet(original_alexnet):
    optimized_model = copy.deepcopy(original_alexnet)

    # 因为是对 AlexNet 就不进行特判了

    # 选择内存占用最高的第一层修改 原始 nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
    # 把输出通道数减半 更新 nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2)
    old_conv0 = optimized_model.features[0]
    new_out_channels_conv0 = old_conv0.out_channels // 2
    
    new_conv0 = nn.Conv2d(
        in_channels=old_conv0.in_channels,
        out_channels=new_out_channels_conv0,
        kernel_size=old_conv0.kernel_size,
        stride=old_conv0.stride,
        padding=old_conv0.padding,
        bias=(old_conv0.bias is not None)
    )
    optimized_model.features[0] = new_conv0

    # 由于前面的输出通道数改变，后续的输入通道数也需要改变
    # 原始 model.features[3] = Conv2d(64, 192, kernel_size=5, padding=2)
    # 更新 Conv2d(32, 192, kernel_size=5, padding=2)
    old_conv3 = optimized_model.features[3]
    new_conv3 = nn.Conv2d(
        in_channels=new_out_channels_conv0, # 调整输入通道
        out_channels=old_conv3.out_channels,
        kernel_size=old_conv3.kernel_size,
        padding=old_conv3.padding,
        bias=(old_conv3.bias is not None)
    )
    optimized_model.features[3] = new_conv3
        
    return optimized_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch AlexNet Profiling (Tasks 2 & 3)")
    # 哪个模型
    parser.add_argument(
        "--model", 
        type=str, 
        default="ori", 
        choices=["ori", "opt"],
        help="'ori' for original AlexNet (Task 2), 'opt' for optimized AlexNet (Task 3)."
    )
    # 通用参数
    parser.add_argument(
        "--row_limit", 
        type=int, 
        default=-1,  # 显示 profiler 表格的行数，默认 -1 全部显示
        help="Number of rows to display in profiler tables."
    )
    parser.add_argument(
        "--record_shapes",
        action="store_true",  # 是否记录形状
        help="Record shapes of input tensors."
    )
    parser.add_argument(
        "--sorted",
        action="store_true",  # 显示各个指标排序
        help="Sort profiler results by the specified metric."
    )
    parser.add_argument(
        "--trace_export",
        action="store_true",  # 是否导出 trace 文件
        help="Export profiler results to a trace file."
    )
    # 模型输入参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # 默认 batch size 为 1
        help="Batch size for the model input."
    )
    parser.add_argument(
        "--C",
        type=int,
        default=3,  # 默认输入通道数为 3
        help="Number of input channels for the model."
    )
    parser.add_argument(
        "--H",
        type=int,
        default=224,  # 默认输入高度为 224
        help="Height of the input tensor."
    )
    parser.add_argument(
        "--W",
        type=int,
        default=224,  # 默认输入宽度为 224
        help="Width of the input tensor."
    )

    args = parser.parse_args()

    # 默认 cuda 可用就启用 cuda 的 profiler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 用于之前的 profiler 版本，当前torchprof_xdu 版本不需要
    profiler_activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        profiler_activities.append(ProfilerActivity.CUDA)

    sample_input = torch.randn(args.batch_size, args.C, args.H, args.W)
    print(f"Batch size: {args.batch_size}, Channels: {args.C}, Height: {args.H}, Width: {args.W}")

    if args.model == "ori":
        print("\nRunning Experiment: Original AlexNet Profiling (Task 2)")
        alexnet = models.alexnet(weights=None)  # 因为只是测试速度，所以就不加载预训练权重了
        profile_model(alexnet, "Original AlexNet", sample_input, device, 
                      profiler_activities, args.row_limit, args.record_shapes,
                      profile_memory=True, with_stack=True, with_flops=True, sorted=args.sorted,
                      trace_export=True)  # 是否导出 trace 文件
        
        if args.trace_export:
            trace_file = "original_alexnet_trace.json"
            alexnet.to(device)
            sample_input = sample_input.to(device)
            for _ in range(2):
                _ = alexnet(sample_input)
            with profile(
                activities=profiler_activities,
                record_shapes=args.record_shapes,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                with torch.no_grad():
                    _ = alexnet(sample_input)
            prof.export_chrome_trace(trace_file)
            print(f"\nTrace file exported to: {trace_file}")
    elif args.model == "opt":
        print("\nRunning Experiment: Optimized AlexNet Profiling (Task 3 - Optimization Part)")
        
        alexnet = models.alexnet(weights=None)  # 因为只是测试速度，所以就不加载预训练权重了
        
        print("Creating 'optimized' AlexNet by modifying its structure...")
        optimized_alexnet = create_optimized_alexnet(alexnet)
        profile_model(optimized_alexnet, "Optimized AlexNet", sample_input, device, 
                      profiler_activities, args.row_limit, args.record_shapes,
                      profile_memory=True, with_stack=True, with_flops=True, sorted=args.sorted,
                      trace_export=True)  # 是否导出 trace 文件
        if args.trace_export:
            trace_file = "optimized_alexnet_trace.json"
            optimized_alexnet.to(device)
            sample_input = sample_input.to(device)
            for _ in range(2):
                _ = optimized_alexnet(sample_input)
            with profile(
                activities=profiler_activities,
                record_shapes=args.record_shapes,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                with torch.no_grad():
                    _ = optimized_alexnet(sample_input)
            prof.export_chrome_trace(trace_file)
            print(f"\nTrace file exported to: {trace_file}")