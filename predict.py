import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.R_models import R_RES
from data import get_loader
import matplotlib.pyplot as plt
import os
import collections
import numpy as np
import cv2 as cv
import pickle
from tqdm import tqdm
import time



def process_dict(stage_dict: dict):
    new_stage_dict = collections.OrderedDict()
    for keys in stage_dict.keys():
        new_keys = keys.split('module.')[-1]
        new_stage_dict[new_keys] = stage_dict[keys]
    return new_stage_dict


def predict_img():
        name = 'R_Model'
        file_dict = {1: 'DUTS', 2: 'ECSSD', 3: 'HKU-IS', 4: 'PASCAL-S', 5: 'DUT-OMRON'}
        model = R_RES()
        model.cuda()
        dic = torch.load(os.path.join('./models', name, 'model.pth'))
        model.load_state_dict(process_dict(dic['model']))
        model.eval()
        batch_size = 48
        save_file=name
        print(save_file)
        if not os.path.exists(os.path.join('./predict_result', save_file)):
            os.makedirs(os.path.join('./predict_result', save_file))
        counter = 0
        for key in range(1, 6):
            if key == 1:
                test_loader = get_loader('./data/DUTS-TE/DUTS-TE-Image/', './data/DUTS-TE/DUTS-TE-Mask/',
                                         batchsize=batch_size,
                                         trainsize=352, shuffle=False, mode='test')
            elif key == 2:
                test_loader = get_loader('./data/ECSSD/', './data/ECSSD-GT/', batchsize=batch_size,
                                         trainsize=352, shuffle=False, mode='test')
            elif key == 3:
                test_loader = get_loader('./data/HKU-IS/', './data/HKU-IS-GT/', batchsize=batch_size,
                                         trainsize=352, shuffle=False, mode='test')
            elif key == 4:
                test_loader = get_loader('./data/PASCAL-S/', './data/PASCAL-S-GT/', batchsize=batch_size,
                                         trainsize=352, shuffle=False, mode='test')
            elif key == 5:
                test_loader = get_loader('./data/DUT-OMRON/', './data/DUT-OMRON-GT/', batchsize=batch_size,
                                         trainsize=352, shuffle=False, mode='test')
            else:
                print('key error')
                return None

            if not os.path.exists(os.path.join('predict_result', save_file, file_dict[key])):
                os.mkdir(os.path.join('predict_result', save_file, file_dict[key]))

            with torch.no_grad():
                for i, pack in tqdm(enumerate(test_loader)):
                    images, _, names, shape = pack
                    images = Variable(images)
                    images = images.cuda()
                    atts, dets = model(images)
                    dt = dets.sigmoid().data.cpu().numpy().squeeze()
                    for i in range(len(dt)):
                        counter += 1
                        map = np.around(dt[i] * 255)
                        cv.imwrite(os.path.join('predict_result', save_file, file_dict[key],
                                                names[i].split('.')[0] + '.png'), map)
            del test_loader


if __name__ == '__main__':
    start = time.time()
    predict_img()
    print(time.time() - start)
