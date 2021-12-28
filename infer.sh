WORKING_DIR=/dataset/f1d6ea5b/wenjiaxin/lot


# CKPT_PATH=lot_large_esc_0828_data/checkpoint-5900
CKPT_PATH=1006_lot_large_1.4G/checkpoint-40000
LOAD_PATH=${WORKING_DIR}/results/$CKPT_PATH

INPUT_PATH=${WORKING_DIR}/eva_test/test
OUTPUT_PATH=${LOAD_PATH}/1006_infer_result/


python code/infer.py \
    --load_path $LOAD_PATH \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --enc_seq_length 128 \
    --dec_seq_length 128 \
    --batch_size 32 \
    --device 0