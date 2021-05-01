import os
import os.path as op
from src.params import path_to_data
import retro
import numpy as np

subject = 'sub-01'

sess = 'ses-shinobi_004'

file = os.listdir(op.join(path_to_data, 'shinobi', 'sourcedata', subject, sess))[0]
filepath = op.join(path_to_data, 'shinobi', 'sourcedata', subject, sess, file)

env = retro.make('ShinobiIIIReturnOfTheNinjaMaster-Genesis', state='Level5')

allreps_frames = []
for i in range(10):
    rep_frames = []
    key_log = retro.Movie(filepath)
    env.reset()
    run_completed = False
    while key_log.step():
        a = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
        frame,c,done,i = env.step(a)
        rep_frames.append(frame.flatten())
    rep_frames_flat = np.array(rep_frames).flatten()
    allreps_frames.append(rep_frames_flat)

for rep_frames1 in allreps_frames:
    for rep_frames2 in allreps_frames:
        print(np.array_equal(rep_frames1,rep_frames2))
