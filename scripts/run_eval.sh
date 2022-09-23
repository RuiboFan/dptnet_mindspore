#!/bin/bash

if [ $# != 1 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "Usage: bash run_eval.sh [DEVICE_ID]"
  bash "run_eval.sh 0 weights/rbpn.ckpt"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_SIZE=1

python evaluate.py  --device_id=$1  > eval.log 2>&1 &