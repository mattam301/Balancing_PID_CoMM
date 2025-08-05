echo =======================

for loss_mask in 100 010 011 101 110 111
do
    echo === Running for loss-mask: $loss_mask ===
    for iter in {1..10}
    do
        echo --- Iteration $iter ---
        python -u train.py --lr 0.0001 --batch-size 64 --epochs 150 --temp 2 --Dataset 'IEMOCAP' --loss-mask $loss_mask
    done
done