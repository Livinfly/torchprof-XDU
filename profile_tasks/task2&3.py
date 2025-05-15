import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import ProfilerActivity
# from ..torchprof_xdu.torchprof_xdu_profile_detailed import ProfileDetailed
import argparse
import copy
from utils import profile_model, trace_export_func


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
    parser.add_argument(
        "--use_cpu",
        action="store_true",  # 是否使用 CPU
        help="Use CPU for profiling."
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
    if args.use_cpu: 
        device = torch.device("cpu")
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
            trace_file = f"original_alexnet_{device.type}_trace.json"
            trace_export_func(trace_file, alexnet, sample_input, device, profiler_activities, args.record_shapes)
        
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
            trace_file = f"optimized_alexnet_{device.type}_trace.json"
            trace_export_func(trace_file, optimized_alexnet, sample_input, device, profiler_activities, args.record_shapes)