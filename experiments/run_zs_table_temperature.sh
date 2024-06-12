#!/bin/bash

# Define the base command
BASE_COMMAND="python table_zs.py --split test --temperature"

# Define the temperatures to iterate over
TEMPERATURES=(0.1 0.3 0.5 0.7 0.9)

# Define the options for --modality
MODALITY_OPTIONS=('table' 'context' 'reference')

# Define the seeds
SEEDS=(5731 6824 6986 4271 3533)

# Define the log file
LOG_FILE="output_zs_table_temperature_llama70b_test_5seeds_None.log"

# Function to tag and log messages
log_message() {
    echo "[USER] $1" >> $LOG_FILE
}

# # Iterate over each --modality option
# for modality_option in "${MODALITY_OPTIONS[@]}"; do
#     log_message "Running command with --modality option: $modality_option"
    
#     # Iterate over each temperature
#     for temp in "${TEMPERATURES[@]}"; do
#         log_message "  Running command for temperature: $temp"
        
#         # Iterate over each seed
#         for (( i=0; i<${#SEEDS[@]}; i++ )); do
#             log_message "    Current iteration: $((i+1)), Using seed: ${SEEDS[$i]}"
            
#             # Update the command with the current seed
#             COMMAND="$BASE_COMMAND $temp --use neighboring --modality $modality_option --seed ${SEEDS[$i]}"
            
#             # Execute the command and log the output
#             log_message "        $COMMAND"
#             $COMMAND >> $LOG_FILE 2>&1
#         done
#     done
# done

# log_message "All iterations completed. Check $LOG_FILE for outputs."


# 'None' Modality
log_message "Running All iterations without modality."

python table_zs.py --split test > "$LOG_FILE" 2>&1

# Define the base command
BASE_COMMAND="python table_zs.py --split test --temperature"

# Function to tag and log messages
log_message() {
    echo "[USER] $1" >> $LOG_FILE
}

# Iterate over each temperature
for temp in "${TEMPERATURES[@]}"; do
    log_message "Running command for temperature: $temp"
    
    # Iterate over each seed
    for (( i=0; i<${#SEEDS[@]}; i++ )); do
        log_message "    Current iteration: $((i+1)), Using seed: ${SEEDS[$i]}"
        
        # Update the command with the current seed
        COMMAND="$BASE_COMMAND $temp --seed ${SEEDS[$i]}"
        
        log_message "        $COMMAND"
        $COMMAND >> $LOG_FILE 2>&1
    done
done

log_message "All iterations completed. Check $LOG_FILE for outputs."