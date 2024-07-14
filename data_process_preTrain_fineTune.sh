python3 data_process_preTrain.py

for i in {0..9}
do
    python3 data_process_fineTune.py $i
done