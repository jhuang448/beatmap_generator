map_path = "maps/"
mapdata_path = "mapdata/"
level = 'Hard'
data_path = "mapdata/{}/".format(level)

divisor=4
time_interval = 16
time_interval2 = 64

lim = -1#10000

# training parameters
batch_size = 1
num_workers = 1  # fixed
shuffle = True  # fixed
epoch = 1000  # fixed
lr = 0.001
model_choose = 'ConvLstm'

# momentum max & min
train_glob_max = [3.09358387, 0.06165558]
train_glob_min = [ 0., -0.05206194]

# Hard
train_glob_mean = [3.53243368e-01, -2.09375834e-05]
train_glob_std = [0.19857471, 0.01427526]

# predict
dist_multiplier = 1
note_density = {"Easy": 0.3, "Normal": 0.3, "Hard": 0.4}
slider_favor = 0
divisor_favor = [0] * divisor