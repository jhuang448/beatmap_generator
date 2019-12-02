# Modify name fields before running
python map_reader.py [OSU FILE NAME] [FFMPEG PATH]
python rhythm_evaluator.py mapthis.npz saved_rhythm_model saved_rhythm_model_momentums momentum_minmax.npy
python GAN.py rhythm_data.npz