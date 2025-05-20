import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.profiler import ProfilerActivity
import argparse
import time
import os
from utils import ProfileDetailed, profile_model, trace_export_func_task5, get_raw_measure_dict_from_profiler_data, \
    display_extracted_measure_dict, simple_value_formatter

def get_model_instance_task5(model_name_str, pretrained=False, image_resolution_hw=(224,224)):
    print(f"  Loading model: {model_name_str} (pretrained: {pretrained}) for resolution {image_resolution_hw}")
    if model_name_str.lower() == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif model_name_str.lower() == "mobilenet_v3_large":
        return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    elif model_name_str.lower() == "vit_b_16":
        return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None,
                               image_size=image_resolution_hw[0])

def get_image_transform_task5(resolution_tuple_hw):
    return transforms.Compose([
        transforms.Resize(resolution_tuple_hw, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_profiling_dataloader_task5(data_root_path, batch_size, resolution_tuple_hw,
                                   num_workers=0, num_samples_to_use=None):
    transform = get_image_transform_task5(resolution_tuple_hw)
    dataset = None
    print(f"  Attempting to load Tiny ImageNet from: '{data_root_path}' for resolution {resolution_tuple_hw}")
    current_file_path = os.path.abspath(__file__)
    profile_dir = os.path.dirname(current_file_path)
    # project_root = os.path.dirname(profile_dir)
    dataset_actual_path = os.path.join(profile_dir, data_root_path, 'train')
    print(dataset_actual_path)
    if not os.path.isdir(dataset_actual_path):
        print(f"  ERROR: Tiny ImageNet path '{dataset_actual_path}' is not a valid directory. Please provide the correct path.")
        return None
    try:
        full_dataset = datasets.ImageFolder(dataset_actual_path, transform=transform)
        if not full_dataset or len(full_dataset) == 0:
            print(f"  ERROR: Tiny ImageNet dataset at '{dataset_actual_path}' loaded as empty or failed to load.")
            return None
        if num_samples_to_use is not None and num_samples_to_use > 0 and num_samples_to_use < len(full_dataset):
            indices = list(range(min(num_samples_to_use, len(full_dataset))))
            dataset = Subset(full_dataset, indices)
            print(f"  Using the first {len(dataset)} samples from Tiny ImageNet ('{dataset_actual_path}').")
        else:
            dataset = full_dataset
            print(f"  Using all {len(dataset)} available samples from Tiny ImageNet ('{dataset_actual_path}').")
    except Exception as e:
        print(f"  ERROR: Failed to load Tiny ImageNet from '{dataset_actual_path}': {e}.")
        return None
    if dataset is None or len(dataset) == 0: return None # Should be caught above
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0 and torch.cuda.is_available()))
    print(f"  DataLoader created: BS={batch_size}, NumBatches={len(dataloader)}, TotalSamplesInEpoch={len(dataset)}")
    return dataloader

def raw_draw(raw_data):
    # 根据需求，用 get_raw_measure_dict_from_profiler_data 导出，进行处理作图。
    
    # CPU total
    print("\n--- Extracting specific measures from prof.raw() ---")

    if raw_data:
        # --- 演示提取 "CPU total" (平均值) ---
        cpu_total_avg_dict = get_raw_measure_dict_from_profiler_data(
            raw_data,
            "CPU total",
            average_over_calls=True
        )
        display_extracted_measure_dict(cpu_total_avg_dict, "CPU total (Avg/call)",
                                       is_averaged=True, unit_converter=simple_value_formatter,
                                       top_n=10) # 显示前 10 个

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiling (Tasks 5)")
    # 哪个模型
    parser.add_argument(
        "--model",
        type=str,
        default=["resnet50"], # 默认剖析 resnet50
        choices=["resnet50", "mobilenet_v3_large", "vit_b_16"],
        help="resnet50 or mobilenet_v3_large or vit_b_16)"
    )
    # 通用参数
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/tiny-imagenet-200", # 默认数据集路径
        help="data path for the dataset"
    )
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
    # 实验参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # 默认 batch size 为 1
        help="Batch size for the model input."
    )
    parser.add_argument(
        "--upper_simple",
        action="store_true",  # 是否 64x64 上采样至 224x224
        help="Use simple upper limit for the model input."
    )
    parser.add_argument(
        "--save_output",
        action="store_true", #是否保存prof.display
        help="Save output file about prof.dispaly"
    )
    args = parser.parse_args()

    # 默认 cuda 可用就启用 cuda 的 profiler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_cpu: 
        device = torch.device("cpu")
    print(f"Using device: {device}")    

    # 用于之前的 profiler 版本，当前 torchprof_xdu 版本不需要
    profiler_activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        profiler_activities.append(ProfilerActivity.CUDA)

    if args.upper_simple:
        current_resolution_hw = (224, 224)
        print("using upper limit resolution: 224x224")
    else:
        current_resolution_hw = (64, 64)
        print("using simple upper limit resolution: 64x64")
    
    model_name_to_run = args.model
    print(f"\n\n{'='*25} PROFILING MODEL: {model_name_to_run.upper()} {'='*25}")
    
    model_instance = get_model_instance_task5(
        model_name_to_run,
        image_resolution_hw=current_resolution_hw
    )

    current_batch_size = args.batch_size
    experiment_label = (f"Model={model_name_to_run}, Device={device.type}, "
                        f"Res={current_resolution_hw[0]}x{current_resolution_hw[1]}, BS={current_batch_size}")
    print(f"\n  Configuring Experiment: {experiment_label}")

    current_dataloader = get_profiling_dataloader_task5(
        data_root_path=args.data_path,
        batch_size=current_batch_size,
        resolution_tuple_hw=current_resolution_hw,
        num_workers=0,
        num_samples_to_use=None,  # 全部数据集
    )

    # 因为用于数据集的原因，所以特别实现一版 profile
    model_instance.to(device)
    model_instance.eval()

    print(f"    Warm-up: {2} batches from tiny-imagenet-200...")
    with torch.no_grad():
        warmup_count = 0
        for i_w, batch_w_data in enumerate(current_dataloader):
            if warmup_count >= 2:
                break
            inputs_w, _ = batch_w_data
            _ = model_instance(inputs_w.to(device))
            warmup_count += 1
        if warmup_count != 2:
            print("    Warning: DataLoader did not yield any batches for warm-up.")
    
    with ProfileDetailed(
        model_instance,
        enabled=True, # 假设总是启用，或者添加命令行参数控制
        use_cuda=(device.type == "cuda"),
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            profiled_count = 0
            for i_p, batch_p_data in enumerate(current_dataloader):
                if profiled_count >= 1:
                    break
                inputs_p, _ = batch_p_data
                _ = model_instance(inputs_p.to(device))
                profiled_count += 1
            if profiled_count != 1:
                    print("    Warning: DataLoader did not yield any batches for profiling.")
    
    print(f"\n    --- Results for {experiment_label} (tiny-imagenet-200) ---")

    save_file=f"{model_name_to_run}_{device.type}_batch{current_batch_size}_{224 if args.upper_simple else 64}"
    print(prof.display(top_k=-1))
    if args.save_output:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            print(prof.display(top_k=-1))
        output = f.getvalue()
# 保存到文件
        with open(f"{save_file}.txt", "w",encoding="utf-8") as file:
            file.write(output)
        raw_draw(prof.raw(),save_file)


    if args.sorted:
        print(f"\n--- {model_name_to_run} Profiler Results (Sorted by CPU total time) ---")
        print(prof.display(sort_by='CPU total', top_k=args.row_limit))

        if device.type == 'cuda':
            print(f"\n--- {model_name_to_run} Profiler Results (Sorted by CUDA total time) ---")
            print(prof.display(sort_by="CUDA total", top_k=args.row_limit))

        print(f"\n--- {model_name_to_run} Profiler Results (Sorted by CPU Memory Usage) ---")
        print(prof.display(sort_by="CPU Mem", top_k=args.row_limit))

        if device.type == 'cuda':
            print(f"\n--- {model_name_to_run} Profiler Results (Sorted by CUDA Memory Usage) ---")
            print(prof.display(sort_by="CUDA Mem", top_k=args.row_limit))

    if args.trace_export:
        trace_file = f"{model_name_to_run}_{device.type}_trace.json"
        trace_export_func_task5(
            trace_file=trace_file,
            model=model_instance,
            dataloader=current_dataloader,
            device=device
        )

    print("1. 剖析结果（文本格式）已在每次实验后打印。")
    print("2. Chrome trace JSON 文件已根据 --trace_export 参数生成。")
    print("3. 要生成图表 (饼图/条形图, Top-K 耗时层列表):")
    print("   a. 使用 ProfileDetailed 的 prof.raw() 方法获取结构化数据。")
    print("   b. 编写Python脚本（或使用Jupyter Notebook）处理 prof.raw() 的输出。")
    print("   c. 使用 matplotlib, seaborn, pandas 等库进行绘图。")
