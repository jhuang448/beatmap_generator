# Beatmap Generator for osu!

Reference Repo: https://github.com/kotritrona/osumapper

Data: https://pan.baidu.com/s/1xqTLtfbMd7DMc1sWZxGCBw

.osu Format: https://osu.ppy.sh/help/wiki/osu!_File_Formats/Osu_(file_format)

Currently only using: Standard -> Beatmap Pack -> 800-899

## Data structure                       
- "lst" (transformed data): table of [TICK, TIME, NOTE, IS_CIRCLE, IS_SLIDER, IS_SPINNER, IS_SLIDER_END, IS_SPINNER_END, SLIDING, SPINNING, MOMENTUM, ANGULAR_MOMENTUM, EX1, EX2, EX3], length MAPTICKS
- "wav" (wav data): np.array, shape of [len(snapsize), MAPTICKS, 2, fft_size//4]
- "flow" (flow data): table of [TICK, TIME, TYPE, X, Y, IN_DX, IN_DY, OUT_DX, OUT_DY] notes only

### About MOMENTUM & ANGULAR_MOMENTUM
Two new variables defined by that author to capture the information about the moving speed of the mouse.
Probably they are not necessary.

## DONE:
1. Beatmap resources
2. Create maplist: maplist_Normal.txt, maplist_Hard.txt, maplist_Easy.txt
3. use osureader.py to parse the beatmap
4. save data to .npz files (one for each): transformed_data, wav_data, flow_data
5. CRNN, overfitting very fast...
![alt text](ConvLstm_loss.png)

## TODO:

Sorry for bringing up such a big project...Let's see where we can get.

### Data Preparation
1. think more about the input and output
2. add more data... (may run into memory issues...) [Not urgent]
3. extract other features: (Percussive Feature + Harmonic Feature) or Mel

### Network Architecture (generate the 'transformed' data)
1. one network or two (classification + regression) ?

### GAN (generate the 'flow' data)
If too complicated, we can skip this part.
1. understand the GAN code in the reference repo...

### Evaluation Metrics
1. might want to look into metrics for: beat-tracking, transcription, music generation(?)
2. listening test...if possible (probably not)

## Deliverables
- Report
- Demo Video: Generate with one click -> Play:)

