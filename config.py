map_path = "maps/"
mapdata_path = "mapdata/"
level = 'Normal'
data_path = "mapdata/{}/".format(level)

divisor=4
time_interval = 16

lim = 10000

# training parameters
batch_size = 1
num_workers = 1  # fixed
shuffle = True  # fixed
epoch = 1000  # fixed
lr = 0.01
model_choose = 'ConvLstm'