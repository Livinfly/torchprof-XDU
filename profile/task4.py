import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity
# from ..torchprof_xdu.torchprof_xdu_profile_detailed import ProfileDetailed
import argparse
from utils import MultiInputNet, profile_model, trace_export_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch AlexNet Profiling (Tasks 2 & 3)")
    # 哪个模型
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet18",  
        choices=["resnet18", "mobilenetv2", "mim"],  # mim 是 multi-input model
        help="resnet18 or mobilenetv2 or multi-input model"
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

    if args.model != "mim":
        sample_input = torch.randn(args.batch_size, args.C, args.H, args.W)
        print(f"Batch size: {args.batch_size}, Channels: {args.C}, Height: {args.H}, Width: {args.W}")

    if args.model == "resnet18":
        print("\nRunning Experiment: ResNet-18 Profiling")
        resnet18 = models.resnet18(weights=None)  # 因为只是测试速度，所以就不加载预训练权重了
        profile_model(resnet18, "ResNet18", sample_input, device, 
                      profiler_activities, args.row_limit, args.record_shapes,
                      profile_memory=True, with_stack=True, with_flops=True, sorted=args.sorted,
                      trace_export=True)  # 是否导出 trace 文件
        
        if args.trace_export:
            trace_file = "resnet18_trace.json"
            trace_export_func(trace_file, resnet18, sample_input, device, profiler_activities, args.record_shapes)

    elif args.model == "mobilenetv2":
        print("\nRunning Experiment: MobileNetV2 Profiling")
        mobilenetv2 = models.mobilenet_v2(weights=None)  # 因为只是测试速度，所以就不加载预训练权重了
        profile_model(mobilenetv2, "MobileNetV2", sample_input, device, 
                      profiler_activities, args.row_limit, args.record_shapes,
                      profile_memory=True, with_stack=True, with_flops=True, sorted=args.sorted,
                      trace_export=True)
        
        if args.trace_export:
            trace_file = "mobilenetv2_trace.json"
            trace_export_func(trace_file, mobilenetv2, sample_input, device, profiler_activities, args.record_shapes)
        
    elif args.model == "mim":
        print("\nRunning Experiment: Multi-Input Model Profiling")
        input_dim_a, input_dim_b = 128, 64
        hidden_dim_a, hidden_dim_b = 256, 128
        branch_output_a, branch_output_b = 32, 16
        shared_hidden_dim = 64
        num_classes = 10

        mim_model = MultiInputNet(
            input_dim_image=input_dim_a, hidden_dim_image=hidden_dim_a, branch_output_image=branch_output_a,
            input_dim_text=input_dim_b, hidden_dim_text=hidden_dim_b, branch_output_text=branch_output_b,
            shared_hidden_dim=shared_hidden_dim, num_classes=num_classes
        )
        sample_input_a = torch.randn(args.batch_size, input_dim_a)
        sample_input_b = torch.randn(args.batch_size, input_dim_b)
        sample_input = (sample_input_a, sample_input_b)

        print(f"Input A shape: {sample_input_a.shape}, Input B shape: {sample_input_b.shape}")

        profile_model(mim_model, "MultiInputNet", sample_input, device, 
                      profiler_activities, args.row_limit, args.record_shapes,
                      profile_memory=True, with_stack=True, with_flops=True, sorted=args.sorted,
                      trace_export=True)
        
        if args.trace_export:
            trace_file = "mim_model_trace.json"
            trace_export_func(trace_file, mim_model, sample_input, device, profiler_activities, args.record_shapes)