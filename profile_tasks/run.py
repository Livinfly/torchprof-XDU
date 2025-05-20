import os
base_command = f"python .\\profile_tasks\\task5.py "
for model_name in ["resnet50", "mobilenet_v3_large", "vit_b_16"]:
    for batch_size in [1, 8, 32]:
        for is_upper in [0, 1]:
            for device in [0, 1]:
                use_cpu_flag = "--use_cpu" if device > 0 else ""
                upper_flag = "--upper_simple" if is_upper > 0 else ""
                command = (
                    f"{base_command} --model={model_name} --batch_size={batch_size} "
                    f"{upper_flag} {use_cpu_flag}"
                )
                print(command)
                os.system(command)
