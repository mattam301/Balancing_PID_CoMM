echo =======================
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $iter ---
CUDA_VISIBLE_DEVICES=5 python -u train_comm.py --lr 0.0001 --batch-size 64 --epochs 150 --temp 2 --Dataset 'IEMOCAP'
done > sdt_comm_iemocap.txt 

# CUDA_VISIBLE_DEVICES=4 python -u train_comm.py --lr 0.0001 --batch-size 64 --epochs 150 --temp 2 --Dataset 'IEMOCAP' > sdt_comm.txt
