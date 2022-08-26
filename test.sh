python test.py \
--resume './outputs/train_IRSENet_1/checkpoints/model_399.pt' \
--experiment_name 'test_IRSENet_1_399' \
--data_root '../../Data/Data_ArrangeRecon/' \
--model_type 'model_cnn' \
--net_G 'DuRDN' \
--norm 'BN' \
--n_filters 16 \
--growth_rate 16 \
--n_denselayer 3 \
--n_channels 32 \
--eval_epochs 5 \
--snapshot_epochs 5 \
--num_workers 0 \
--gpu_ids 0

