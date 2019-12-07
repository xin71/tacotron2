#!/bin/bash
python process_spec.py
python train.py --output_directory="./outputs" --log_directory="./logs"

