#!/bin/bash
uname -a
#date
#env
date

DATASET=voc
DATA_PATH=YOUR_PASCAL_PATH
TRAIN_LIST=./dataset/list/voc/trainaug.txt
VAL_LIST=./dataset/list/voc/val.txt
FOLD=0
SHOT=1
MODEL=pspnet_pop
BACKBONE=resnet50v2
RESTORE_PATH=YOUR_RESTORE_PATH
BS=1
BASE_SIZE=473,473
OS=8
SAVE=0
SAVE_DIR=YOUR_SAVE_DIR
SEED=123,234

cd YOUR_CODE_DIR
python eval_ft.py  --dataset ${DATASET} --data-dir ${DATA_PATH} \
                --train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --test-batch-size ${BS} \
                --model ${MODEL} --restore-from ${RESTORE_PATH} --backbone ${BACKBONE} \
                --base-size ${BASE_SIZE} --save-path ${SAVE_DIR} --save ${SAVE}\
                --fold ${FOLD} --shot ${SHOT} --os ${OS} --random-seed ${SEED}