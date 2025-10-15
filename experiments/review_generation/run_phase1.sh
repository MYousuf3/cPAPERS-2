#!/bin/bash

# Run Phase 1: Data Preparation
# This script splits the dataset and prepares training data

echo "=========================================="
echo "Phase 1: Data Preparation"
echo "=========================================="

cd "$(dirname "$0")"

python3 phase1_data_preparation.py

echo ""
echo "Phase 1 complete!"
echo "Check the output in ../../data/processed/"

