echo =======================
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $iter ---
CUDA_VISIBLE_DEVICES=3 python -u train.py --lr 0.0001 --batch-size 64 --epochs 150 --temp 2 --Dataset 'OLD_IEMOCAP' --loss-mask 111
# CUDA_VISIBLE_DEVICES=3 python -u train.py --lr 0.0001 --batch-size 64 --epochs 150 --temp 2 --Dataset 'MELD' --loss-mask 111
done 

# CUDA_VISIBLE_DEVICES=4 python -u train_comm.py --lr 0.0001 --batch-size 64 --epochs 150 --temp 2 --Dataset 'IEMOCAP' > sdt_comm.txt
# python -u train.py --lr 0.0001 --batch-size 64 --epochs 300 --temp 2 --Dataset 'MELD' --loss-mask 111
