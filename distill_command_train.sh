CUDA_VISIBLE_DEVICES=0 python train_distill.py --camera realsense --log_dir logs/log_rs --num_pts_stu 15000 --batch_size 2 --dataset_root /data/datasets/graspnet --model_name graspness_Distill_stu15k_rs --max_epoch 12 --resume 0 --distillation 1 --teacher_ckpt logs/log_rs_objpts/graspness_15k_rs_C_epoch20.tar

