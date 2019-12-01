map_path = "maps/"
mapdata_path = "mapdata/"
level = 'Normal'
data_path = "mapdata/{}/".format(level)

divisor=4
time_interval = 16
time_interval2 = 64

lim = 10000

# training parameters
batch_size = 1
num_workers = 1  # fixed
shuffle = True  # fixed
epoch = 1000  # fixed
lr = 0.001
model_choose = 'ConvLstm2'

# momentum max & min
train_glob_max = [3.09358387, 0.06165558]
train_glob_min = [ 0., -0.05206194]
train_glob_mean = [0, 0]
train_glob_std = [0, 0]

# predict
dist_multiplier = 1
note_density = 0.36
slider_favor = 0
divisor_favor = [0] * divisor