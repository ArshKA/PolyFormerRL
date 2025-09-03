#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6061

det_weight=0.1
cls_weight=0.0005
num_bins=64
log_dir=/data0/arshkon/checkpoints/polyform_rl/polyformer_l_logs
save_dir=/data0/arshkon/checkpoints/polyform_rl/polyformer_l_checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../polyformer_module

data_dir=/data0/arshkon/checkpoints/polyform_rl/datasets/finetune
data=${data_dir}/refcoco+g_train_shuffled.tsv,${data_dir}/refcoco/refcoco_val.tsv
selected_cols=0,5,6,2,4,3,7
restore_file=/data0/arshkon/checkpoints/polyform_rl/polyformer_l_pretrain.pt
train_tsv=${data_dir}/refcoco+g_train_shuffled.tsv

task=refcoco
arch=polyformer_l
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=5
warmup_ratio=0.06
batch_size=4
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=420

patch_image_size=512

WORLD_SIZE=8

for max_epoch in 100; do
  echo "max_epoch "${max_epoch}
  for lr in 5e-5; do
    echo "lr "${lr}
    for patch_image_size in 512; do
      echo "patch_image_size "${patch_image_size}

      log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
      save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
      mkdir -p $save_path

      # compute total updates and warmup-updates from warmup_ratio
      n_train=$(wc -l < "${train_tsv}")
      effective_per_update=$(( batch_size * update_freq * WORLD_SIZE ))
      updates_per_epoch=$(( (n_train + effective_per_update - 1) / effective_per_update ))
      total_num_update=$(( max_epoch * updates_per_epoch ))
      warmup_updates=$(python - <<PY
ratio = float("${warmup_ratio}")
total = int(${total_num_update})
print(int(ratio * total))
PY
)

      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=${WORLD_SIZE} --master_port=${MASTER_PORT} --standalone ../../train.py \
          $data \
          --selected-cols=${selected_cols} \
          --bpe-dir=${bpe_dir} \
          --user-dir=${user_dir} \
          --reset-optimizer --reset-dataloader --reset-meters \
          --save-dir=${save_path} \
          --task=${task} \
          --arch=${arch} \
          --criterion=${criterion} \
          --label-smoothing=${label_smoothing} \
          --batch-size=${batch_size} \
          --update-freq=${update_freq} \
          --encoder-normalize-before \
          --restore-file=${restore_file} \
          --decoder-normalize-before \
          --share-decoder-input-output-embed \
          --share-all-embeddings \
          --layernorm-embedding \
          --patch-layernorm-embedding \
          --code-layernorm-embedding \
          --resnet-drop-path-rate=${resnet_drop_path_rate} \
          --encoder-drop-path-rate=${encoder_drop_path_rate} \
          --decoder-drop-path-rate=${decoder_drop_path_rate} \
          --dropout=${dropout} \
          --attention-dropout=${attention_dropout} \
          --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay --lr=${lr} \
          --max-epoch=${max_epoch} --total-num-update=${total_num_update} --warmup-updates=${warmup_updates} \
          --log-format=simple --log-interval=10 \
          --fixed-validation-seed=7 \
          --no-epoch-checkpoints --keep-best-checkpoints=1 \
          --save-interval=1 --validate-interval=1 \
          --save-interval-updates=500 --validate-interval-updates=500 \
          --eval-acc \
          --eval-args='{"beam":5,"min_len":2,"max_len_a":0,"max_len_b":2}' \
          --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --patch-image-size=${patch_image_size} \
          --fp16 \
          --fp16-scale-window=512 \
          --det_weight=${det_weight} \
          --cls_weight=${cls_weight} \
          --num-workers=0 > ${log_file} 2>&1
    done
  done
done
