pip install --upgrade turibolt --index https://pypi.apple.com/simple
pip install -r requirements.txt

# download unsup data
sh data/download_wiki.sh

cd SentEval/data/downstream/
bash download_dataset.sh

cd ../../..

# run train
python train.py \
    --model_name_or_path roberta-large \
    --train_file wiki1m_for_simcse.txt \
    --output_dir ${BOLT_ARTIFACT_DIR}/simcse-roberta-large \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --negative_dropout_rate 0.5 \
    --negative_dropout

python simcse_to_huggingface.py --path ${BOLT_ARTIFACT_DIR}/baseline-simcse-bert-base-uncased

# run eval
echo "START EVALUATION"
python evaluation.py \
	--model_name_or_path ${BOLT_ARTIFACT_DIR}/baseline-simcse-bert-base-uncased \
	--pooler cls_before_pooler \
	--task_set full \
	--mode test


