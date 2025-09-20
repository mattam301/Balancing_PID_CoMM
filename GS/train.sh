# for seed in {1..20}; do
#   echo "Running with seed $seed"
#   python -u run.py \
#     --gpu 1 \
#     --port 1531 \
#     --classify emotion \
#     --dataset CMUMOSEI7 \
#     --epochs 120 \
#     --textf_mode textf0 \
#     --loss_type emo_sen_sft \
#     --lr 1e-04 \
#     --batch_size 10 \
#     --hidden_dim 512 \
#     --win 17 17 \
#     --heter_n_layers 7 7 7 \
#     --drop 0.2 \
#     --shift_win 19 \
#     --lambd 1.0 1.0 0.7 \
#     --real_validation \
#     --seed $seed
# done > graph_smile_mosei.txt


for seed in {1..20}; do
  echo "Running with seed $seed"
  python -u run.py \
    --gpu 1 \
    --port 1531 \
    --classify emotion \
    --dataset MELD \
    --epochs 120 \
    --textf_mode textf0 \
    --loss_type emo_sen_sft \
    --lr 1e-04 \
    --batch_size 10 \
    --hidden_dim 512 \
    --win 17 17 \
    --heter_n_layers 7 7 7 \
    --drop 0.2 \
    --shift_win 19 \
    --lambd 1.0 1.0 0.7 \
    --real_validation \
    --seed $seed
done > graph_smile_meld_2.txt