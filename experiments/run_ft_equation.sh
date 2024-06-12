#!/bin/bash

# Define the log file
LOG_FILE="ft_equation.log"

# Function to log command and output to file
log_command() {
    echo "Command: $1" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
    eval "$1" >> "$LOG_FILE" 2>&1
    echo "" >> "$LOG_FILE"
}

# Define the temperature values
temperatures=(0.1 0.3 0.5 0.7 0.9)

# Define the seed values
SEEDS=(5731 6824 6986 4271 3533)

# Log another initial command
log_command "python equation_ft.py --train"
log_command "python equation_ft.py --eval --split 'test'"
# Loop through each temperature value
for temp in "${temperatures[@]}"
do
    # Log command for each seed iteration
    for seed in "${SEEDS[@]}"
    do
        log_command "python equation_ft.py --eval --split 'test' --temperature $temp --seed $seed"
    done
done

# Log initial command
log_command "python equation_ft.py --train --modality 'equation'"
log_command "python equation_ft.py --eval --modality 'equation' --split 'test'"
# Loop through each temperature value
for temp in "${temperatures[@]}"
do
    # Log command for each seed iteration
    for seed in "${SEEDS[@]}"
    do
        log_command "python equation_ft.py --eval --modality 'equation' --split 'test' --temperature $temp --seed $seed"
    done
done


# Log another initial command
log_command "python equation_ft.py --train --modality 'context'"
log_command "python equation_ft.py --eval --modality 'context' --split 'test'"
# Loop through each temperature value
for temp in "${temperatures[@]}"
do
    # Log command for each seed iteration
    for seed in "${SEEDS[@]}"
    do
        log_command "python equation_ft.py --eval --modality 'context' --split 'test' --temperature $temp --seed $seed"
    done
done

# Log another initial command
log_command "python equation_ft.py --train --modality 'reference'"
log_command "python equation_ft.py --eval --modality 'reference' --split 'test'"
# Loop through each temperature value
for temp in "${temperatures[@]}"
do
    # Log command for each seed iteration
    for seed in "${SEEDS[@]}"
    do
        log_command "python equation_ft.py --eval --modality 'reference' --split 'test' --temperature $temp --seed $seed"
    done
done