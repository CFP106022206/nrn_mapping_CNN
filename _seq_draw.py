import os, shutil
import logging
from datetime import datetime
import pytz
import pandas as pd

# Primary params
FILE_PATH = "data/selected_data/test_10to20.csv" # CSV to read and run, with full path from the root folder
NEW_DIR = 'three_view_pic_rk10to20/' # !!! Save pkls in 'statistical_results/' + NEW_DIR. Need to add '/' at the end
W_KEY = 'sn' # Weighting keys: unit, sn, rsn. Choose ONE for per run

# Not necessary to change
PREFIX = '_' # Saved filename: 'mapping_data_' + W_KEY + PREFIX + fc_id + '.pkl'
CALL_PY = 'main_draw.py'

def get_time():
    tz = pytz.timezone("Asia/Taipei") 
    Time = datetime.now(tz)
    return Time

# Initializing folders for sequentially run 'main_all.py'
try:
    os.mkdir('data/selected_data/TEMP_FC/')
    os.mkdir('data/selected_data/TEMP_EM/')
except:
    pass

LOGGING_FORMAT = '%(asctime)s  %(message)s'
DATE_FORMAT = '%y-%m-%d %H:%M:%S'
# logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
logging.basicConfig(level=logging.DEBUG, filename='draw.log', filemode='w', format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
# filemode = a: append ; w: overwrite

logging.Formatter.converter = lambda *args: datetime.now(tz=pytz.timezone('Asia/Taipei')).timetuple()

logging.info('Start Computing...')
T = get_time()

df = pd.read_csv(FILE_PATH)

# Convert single-pair list to fc-ems maps

fc_lst = df['fc_id'].drop_duplicates().to_list()

pairs = []
for fc in fc_lst:
    a = []
    a.append(fc)
    tdf = df[df['fc_id']==fc]
    em_ids = tdf['em_id'].to_list()
    for em in em_ids:
        a.append(str(em))
    pairs.append(a)

total = len(pairs)
count = 1

# Folders for FC files
fc_inlst = os.listdir('data/selected_data/FC')
fc_inlst_a = os.listdir('data/selected_data/FC_add')

# Start to draw (call 'main_all.py' for each fc_id)

for p in pairs:
    ts = get_time()
    fc_id = p[0]
    em_ids = p[1:]

    logging.info(str(count) + ' / ' + str(total))
    
    path = 'data/selected_data/'

    if fc_id + '.swc' in fc_inlst:
        try:
            shutil.copy2(path + 'FC/'+ fc_id + '.swc', path + 'TEMP_FC/' + fc_id + '.swc')
        except:
            s = 'ERROR: ' + fc_id
            logging.info(s)
            continue
    elif fc_id + '.swc' in fc_inlst_a:
        try:
            shutil.copy2(path + 'FC_add/'+ fc_id + '.swc', path + 'TEMP_FC/' + fc_id + '.swc')
        except:
            s = 'ERROR: ' + fc_id
            logging.info(s)
            continue
    else:
        s = 'NOT FOUND: ' + fc_id + '.swc'
        logging.info(s)
        continue # Without mapping EMs, directly go to next fc_id

    for em_id in em_ids:
        try:
            shutil.copy2(path + 'EM/'+ em_id + '.swc', path + 'TEMP_EM/' + em_id + '.swc')
        except:
            s = 'NOT FOUND: ' + em_id + '.swc'
            logging.info(s)
            continue # Go to next em_id

    # Call 'main_all.py'
    s = ['python3 -u', CALL_PY, W_KEY]
    os.system(' '.join(s))

    ### Current path: '/' (the root)
    # Move mapping results in 'statistical_results/' to NEW_DIR
    path = 'data/statistical_results/'
    map_lst = os.listdir(path)
    try:
        os.mkdir(path + NEW_DIR)
    except:
        pass
    for f in map_lst:
        if 'mapping_data' in f:
            os.rename(path + f,
                      path + NEW_DIR + f.split('.pkl')[0] + PREFIX + str(fc_id) + '.pkl')

    tf = get_time()
    d = str(tf - ts)[:-7]
    d_log = '   Duration: ' + d
    logging.info(d_log)
    print()

    # Clear files in TEMP_FC and TEMP_EM for next fc_id
    try:
        os.system('rm data/selected_data/TEMP_FC/*')
    except:
        pass
    try:
        os.system('rm data/selected_data/TEMP_EM/*')
    except:
        pass
    
    count += 1

logging.info('Finished!')
d = '   Total Duration: ' + str(tf-T)
logging.info(d)