import os
import os.path as op
from src.params import path_to_data

subject = 'sub-01'

sess = 'ses-shinobi_004'

file = os.listdir(op.join(path_to_data, 'shinobi', 'sourcedata', subject, sess))[0]
filepath = op.join(path_to_data, 'shinobi', 'sourcedata', subject, sess, file)


run_variables = {}
key_log = retro.Movie(filepath)
env.reset()
run_completed = False
#while key_log.step():

a = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
b,c,done,i = env.step(a)
