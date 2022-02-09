import os.path as op

SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-06']
LEVELS = ['1', '4', '5']
ACTIONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
GAME_FS = 60

SRC_PATH = op.join('.')
DATA_PATH = op.join(SRC_PATH, 'data')
FIG_PATH = op.join(SRC_PATH, 'reports', 'figures')
TABLE_PATH = op.join(SRC_PATH, 'reports', 'tables')
