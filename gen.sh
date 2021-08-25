WORKING_DIR=/dataset/f1d6ea5b/wenjiaxin/lot


CKPT_PATH=test_small_esc/checkpoint-62
LOAD_PATH=${WORKING_DIR}/results/$CKPT_PATH

INPUT_PATH=${WORKING_DIR}/esc/test.source
OUTPUT_PATH=${LOAD_PATH}/generate.txt


python code/gen.py \
    --load_path $LOAD_PATH \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --enc_seq_length 128 \
    --dec_seq_length 128 \
    --batch_size 64 \
    --device 0