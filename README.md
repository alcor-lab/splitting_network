# splitting_network
#It requires the following packages:
Numpy
PIL
torch
torchvision
tqdm
matplotlib
joblib
pynvml

#Example of run 20vs680:
CUDA_VISIBLE_DEVICES=0 python parallel_big_train.py --seed 1 --data_dir /media/Data2/avi_kinetics700_256/ --start_from 34 --no_parallel --gpu 0 --absolute_path --fix_imbalance --epochs 1

#example of training all 34 splits:
python parallel_big_train.py --seed 1 --data_dir ~/softlinkFolder20vs20/ --start_from 0

#to extract prediction from one single split:
CUDA_VISIBLE_DEVICES=0 python parallel_big_train.py --seed 1 --test  --data_dir /media/Data2/avi_kinetics700_256_validation/ --start_from 34 --no_parallel --gpu 0 --absolute_path

#to aggregate predictions to get the final result:
python aggregate_predictions.py --seed 1 --test --data_dir /media/Data2/avi_kinetics700_256_validation/ --start_from 0 --end_to 35
