echo =======================

for loss_mask in 11111 11110
do
    echo === Running for loss-mask: $loss_mask ===
    for iter in {1..10}
    do
        echo --- Iteration $iter ---
        CUDA_VISIBLE_DEVICES=5 python -u train_comm.py --lr 0.0001 --batch-size 64 --epochs 200 --temp 2 --Dataset 'IEMOCAP' --loss_mask $loss_mask
    done
done