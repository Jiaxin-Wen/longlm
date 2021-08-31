DATA_PATH=esc_manual
# DATA_PATH=data
LOAD_PATH=/dataset/f1d6ea5b/wenjiaxin/lot/results/lot_large_esc_0828_data/checkpoint-5900
SAVE_DIR=lot_large_manual0831/

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
HOST=m0
HOST_FILE=config/hostfile/hostfile-$HOST

OPTS=""
OPTS+="--data_dir $DATA_PATH"
OPTS+=" --train_name train"
OPTS+=" --max_target_length 128"
OPTS+=" --max_source_length 128"
OPTS+=" --val_max_target_length 128"
OPTS+=" --test_max_target_length 128"
OPTS+=" --output_dir results/${SAVE_DIR}"
OPTS+=" --save_total_limit 10"
OPTS+=" --per_device_train_batch_size 4"
# OPTS+=" --per_device eval_batch size 4"
OPTS+=" --num_train_epochs 2"
OPTS+=" --logging_steps 5"
OPTS+=" --model_name_or_path $LOAD_PATH"
OPTS+=" --learning_rate 1e-4"
OPTS+=" --n_val 1000"
# OPTS+=" --evaluation_strategy steps"
OPTS+="  --evaluation_strategy no"
OPTS+=" --eval_steps 100"
OPTS+=" --do_train"
# OPTS+=" --do_eval"
# OPTS+=" --overwrite_output_dir"
OPTS+=" --load_best_model_at_end"
OPTS+=" --gradient_accumulation_steps 1"
OPTS+=" --deepspeed config/deepspeed/ds_zero2_config_st.json"


export NCCL_DEBUG=INFO
CMD="/opt/conda/bin/deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port 4586 --hostfile ${HOST_FILE} code/finetune_trainer.py $@ $OPTS"
echo ${CMD}
${CMD}
    # --data_dir=$DATA_PATH \
    # --train_name=train \
    # --max_target_length 128 \
    # --max_source_length 128 \
    # --val_max_target_length 128 \
    # --test_max_target_length 128 \
    # --output_dir=results/$SAVE_DIR \
    # --save_total_limit=10 \
    # --per_device_train_batch_size=4 \
    # # --per_device_eval_batch_size=4 \
    # --num_train_epochs=2 \
    # --logging_steps=5 \
    # --model_name_or_path=$LOAD_PATH \
    # --learning_rate=1e-4 \
    # --n_val=1000 \
    # --evaluation_strategy=steps \
    # --eval_steps=100 \
    # --do_train \
    # # --do_eval \
    # # --overwrite_output_dir \
    # --load_best_model_at_end \
    # --gradient_accumulation_steps 1 \
    # --deepspeed config/deepspeed/ds_zero2_config_st.json
