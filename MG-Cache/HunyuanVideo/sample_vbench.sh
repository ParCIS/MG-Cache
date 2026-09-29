#!/bin/bash

devices=(
    "4"
)
Num_Devices=${#devices[@]}

target_item="mg-cache"

JSON_FILE="./eval/VBench_sub_info.json"

# video save path
Video_Save_Path="./eval/${target_item}"
Log_Path="./eval/log/${target_item}"

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found at ${JSON_FILE}"
    exit 1
fi

mkdir -p "${Log_Path}"

# Calculate the total number of prompts in the JSON file
total_prompts=$(python3 -c "import json; f=open('$JSON_FILE'); data=json.load(f); print(len(data))")

# Compute the number of prompts each device will handle (the last device may have fewer)
chunk_size=$(( (total_prompts + Num_Devices - 1) / Num_Devices ))

# Launch separate background processes for each GPU
for i in "${!devices[@]}"; do
    {
        index_start=$(( i * chunk_size ))
        index_end=$(( (i+1) * chunk_size))
        if [ $index_end -ge $total_prompts ]; then
            index_end=$(( total_prompts))
        fi
        
        log_file="${Log_Path}/chunk_${i}.log"
        echo "Device ${devices[$i]}: Processing prompts index range [$index_start, $index_end]"
        
        CUDA_VISIBLE_DEVICES="${devices[$i]}" python3 sample_video_vbench.py \
            --vbench-json-path "$JSON_FILE" \
            --index-start "$index_start" \
            --index-end "$index_end" \
            --num-per-prompt 5 \
            --seed 42 \
            --video-size 480 640\
            --video-length 65 \
            --infer-steps 50 \
            --flow-reverse \
            --use-cpu-offload \
            --alpha 0.0 \
            --beta 0.0 \
            --save-path "$Video_Save_Path" >> "$log_file" 2>&1
        
        echo "Device ${devices[$i]}: Completed inference for index range [$index_start, $index_end]"
    }&
    sleep 10
done

wait
echo "All GPU processes have completed."