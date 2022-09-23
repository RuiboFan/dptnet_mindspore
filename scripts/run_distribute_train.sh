#!/bin/bash


if [ $# != 2 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE]"
  echo "bash run_distribute_train.sh 8 1 ./hccl_8p.json"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
fi

export RANK_TABLE_FILE=$2
export RANK_START_ID=0
export RANK_SIZE=$1
echo "lets begin!!!!XD"

for((i=0;i<$1;i++))
do
        export DEVICE_ID=$((i + RANK_START_ID))
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"

        python train.py --device_num=$1 > parallel_train$i.log 2>&1 &
done
