import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re
import json
import sys

def read_npz(fn):
    with np.load(fn) as data:
        wav_data = data["wav"]
        wav_data = np.swapaxes(wav_data, 2, 3)
        ticks = data["ticks"]
        timestamps = data["timestamps"]
        extra = data["extra"]
        
        # Extra vars
        bpms = extra[0]
        slider_lengths = extra[1]
        ex1 = (60000 / bpms) / 500 - 1
        ex2 = bpms / 120 - 1
        ex3 = slider_lengths / 150 - 1
        
        div_data = np.array([divisor_array(k) + [ex1[k], ex2[k], ex3[k]] for k in ticks]);
    return wav_data, div_data, ticks, timestamps

def divisor_array(k):
    d_range = list(range(0, divisor))
    return [int(k % divisor == d) for d in d_range]

def load_momentum_minmax(fn):
    data = np.load(fn)
    return data

"""
    Inputs:
        model_name
        mapthis npz
        momentum_minimax.npy
        saved_rhythm_model_momentyms
"""

# argument
dist_multiplier = 1
note_density = 0.36
slider_favor = 0
divisor = 4
divisor_favor = [0] * divisor

def predict_rhythm_momumtum(mapthis_file, rhythm_model_name, momentum_model_name, momentum_minimax):
    model = tf.keras.models.load_model(
        rhythm_model_name,
        custom_objects=None,
        compile=False
    )
    model.compile(loss='mse',
            optimizer=tf.train.RMSPropOptimizer(0.001),
            metrics=[keras.metrics.mae])

    test_data, div_data, ticks, timestamps = read_npz(mapthis_file)

    # Make time intervals from test data
    time_interval = 16
    if test_data.shape[0]%time_interval > 0:
        test_data = test_data[:-(test_data.shape[0]%time_interval)]
        div_data = div_data[:-(div_data.shape[0]%time_interval)]
    test_data2 = np.reshape(test_data, (-1, time_interval, test_data.shape[1], test_data.shape[2], test_data.shape[3]))
    div_data2 = np.reshape(div_data, (-1, time_interval, div_data.shape[1]))

    test_predictions = model.predict([test_data2, div_data2])
    preds = test_predictions.reshape(-1, test_predictions.shape[2])

    # Favor sliders a little
    preds[:, 2] += slider_favor
    divs = div_data2.reshape(-1, div_data2.shape[2])
    margin = np.sum([divisor_favor[k] * divs[:, k] for k in range(0, divisor)])

    preds[:, 0] += margin

    # Predict is_obj using note_density
    obj_preds = preds[:, 0]
    target_count = np.round(note_density * obj_preds.shape[0]).astype(int)
    borderline = np.sort(obj_preds)[obj_preds.shape - target_count]
    is_obj_pred = np.expand_dims(np.where(preds[:, 0] > borderline, 1, 0), axis=1)

    obj_type_pred = np.sign(preds[:, 1:4] - np.tile(np.expand_dims(np.max(preds[:, 1:4], axis=1), 1), (1, 3))) + 1
    others_pred = (1 + np.sign(preds[:, 4:test_predictions.shape[1]] + 0.5)) / 2
    another_pred_result = np.concatenate([is_obj_pred, is_obj_pred * obj_type_pred, others_pred], axis=1)

    print("{} notes predicted.".format(np.sum(is_obj_pred)))


    mommax, mommin = load_momentum_minmax(momentum_minimax)

    momentum_model = tf.keras.models.load_model(
        momentum_model_name,
        custom_objects=None,
        compile=False
    )
    momentum_model.compile(loss='mse',
                optimizer=tf.train.RMSPropOptimizer(0.001),
                metrics=[keras.metrics.mae])
    
    momentum_predictions_output = momentum_model.predict([test_data2, div_data2])
    momentum_predictions = (momentum_predictions_output.reshape(-1, 2) + 1) / 2 / 0.8 * (mommax - mommin) + mommin

    np.savez_compressed("rhythm_data", objs = is_obj_pred[:, 0], predictions = another_pred_result, timestamps = timestamps, ticks = ticks, momenta = momentum_predictions, sv = (div_data[:,6] + 1) * 150, dist_multiplier = dist_multiplier)
    rhythm_json = {
        "objs": is_obj_pred[:, 0].tolist(), 
        "predictions": another_pred_result.tolist(),
        "timestamps": timestamps.tolist(),
        "ticks": ticks.tolist(),
        "momenta": momentum_predictions.tolist(),
        "sv": ((div_data[:,6] + 1) * 150).tolist(),
        "distMultiplier": dist_multiplier
    }
    with open("evaluatedRhythm.json", "w") as er:
        json.dump(rhythm_json, er)

if __name__ == "__main__":
    mapthis_file = sys.argv[1]
    rhythm_model_name = sys.argv[2]
    momentum_model_name = sys.argv[3]
    momentum_minimax = sys.argv[4]
    predict_rhythm_momumtum(mapthis_file, rhythm_model_name, momentum_model_name, momentum_minimax)



