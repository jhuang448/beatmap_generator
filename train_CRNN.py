import os, torch
from model import ConvLstmNet
from train_utils import Trainer
from lib import Data2Torch, load_data
from config import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # change
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_name = '{}_batch{}_lr{}'.format(model_choose, batch_size, lr)

    # model saving path
    from datetime import date
    date = date.today()
    out_model_fn = './models/%d%d%d/%s/' % (date.year, date.month, date.day, model_name)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    train_data, test_data = load_data()

    # prepare dataloader (function inside lib.py)
    t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': shuffle, 'pin_memory': True,
                'drop_last': True}
    v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    tr_loader = torch.utils.data.DataLoader(Data2Torch(train_data, lim), **t_kwargs)
    va_loader = torch.utils.data.DataLoader(Data2Torch(test_data, lim), **v_kwargs)

    print(len(tr_loader), len(va_loader))

    # build model (function inside model.py)
    model = ConvLstmNet()
    model.to(device)

    # start training (function inside train_utils.py)
    Trer = Trainer(model, lr, epoch, out_model_fn)
    Trer.fit(tr_loader, va_loader, device)

    print(model_name)


if __name__ == "__main__":
    main()