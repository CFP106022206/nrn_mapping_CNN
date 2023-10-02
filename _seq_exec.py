import os
import logging
from datetime import datetime
import pytz

def get_time():
    tz = pytz.timezone("Asia/Taipei") 
    Time = datetime.now(tz)
    return Time

LOGGING_FORMAT = '%(asctime)s  %(message)s'
DATE_FORMAT = '%y-%m-%d %H:%M:%S'
# logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
logging.basicConfig(level=logging.DEBUG, filename='time.log', filemode='w', format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
# filemode =  a: append ; w: overwrite

logging.Formatter.converter = lambda *args: datetime.now(tz=pytz.timezone('Asia/Taipei')).timetuple()

logging.info('Start Calculating...')
T = get_time()

parts = [x+1 for x in range(11)]

for i in parts:
    ts = get_time()
    logging.info('>> Part' + str(i))

    s = 'python -u main.py EM_part' + str(i) + ' > part_em' + str(i) + '.log'
    print('Executing Part'+ str(i)+':', '\''+s+'\'')
    os.system(s)

    # current path: /
    path = 'data/statistical_results/'
    os.rename(path + 'info_list_unit.csv',
            path + 'info_list_unit_em' + str(i) + '.csv')

    tf = get_time()
    d = '   Duration: ' + str(tf-ts)
    logging.info(d)
    print()

logging.info('Finished!')
d = '   Total Duration: ' + str(tf-T)
logging.info(d)