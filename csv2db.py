# %%
### PURPOSE: Create DB file from mapping results (csv) for the webpage
###    NOTE: Make sure that 'rank' column is in the csv file already

import sqlite3
import pandas as pd

csv_path = './result/unlabel_data_predict/2024-01-31_EMxFC/merge_predict_Rank5.csv' # Results from model
DB_FILE = 'DB_Rank5.db' # Target DB filename

df = pd.read_csv(csv_path)
# 重命名列名
df.rename(columns={'model_predict': 'score'}, inplace=True)


# %%## Add 'plot' column for plots filenames

tmpFC = df['fc_id'].to_list()
tmpEM = df['em_id'].to_list()

tmp = []
for i in range(len(tmpFC)):
    tmp.append(str(tmpEM[i]) + '_' + tmpFC[i] + '.png')

df['plot'] = pd.Series(tmp)

### Create DB file

conn = sqlite3.connect(DB_FILE)

conn.execute('CREATE TABLE PAIRS (id int primary key, fc_id varchar(20), em_id varchar(20), score float, rank int, plot varchar(20))')
df.to_sql('PAIRS', conn, if_exists='replace')

conn.close()

# %%
