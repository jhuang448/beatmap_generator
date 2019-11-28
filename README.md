# Beatmap Generator for osu!

reference: https://github.com/kotritrona/osumapper

data: https://pan.baidu.com/s/1xqTLtfbMd7DMc1sWZxGCBw

Currently only using: Standard -> Beatmap Pack -> 800-899

## DONE:
1. Beatmap resources
2. Create maplist: maplist_Normal.txt, maplist_Hard.txt, maplist_Easy.txt

## TODO:

Sorry for bringing up such a big project...Let's see where we can get.

### Data Preparation
1. add more data... (may run into memory issues...)
2. use osureader.py to parse the beatmap
3. extract features: (Percussive Feature + Harmonic Feature) or Mel

### Network Architecture (generate the 'flow' data)
1. figure out what momentum means in the context of beatmap
2. CRNN
3. one network or two (classification + regression) ?

### GAN (use pre-trained model to convert the 'flow' data to a map)
1. understand the GAN code in the reference repo...

### Evaluation Metrics
1. might want to look into metrics for: beat-tracking, transcription, music generation(?)
2. listening test...if possible (probably not)

