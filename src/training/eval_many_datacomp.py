import subprocess
from tqdm.auto import tqdm
import os
import pandas as pd
from model_list import model_list

def image_size_set(model_name, df):
    row = df[df['updated_name'] == model_name]
    if not row.empty:
        imsize = int(row.iloc[0]['image_size'])
        if imsize:
            return imsize
    if "384" in model_name:
        image_size = 384
    elif "512" in model_name:
        image_size = 512
    elif "256" in model_name:
        image_size = 256
    elif "336" in model_name:
        image_size = 336
    elif "280" in model_name:
        image_size = 280
    elif "475" in model_name:
        image_size = 475
    elif "448" in model_name:
        image_size = 448
    else:
        image_size = 224
    return image_size

save_csv = "/scratch/bf996/vlhub/logs/dcc-datacomp-results.csv"

increment = 0

while os.path.exists(save_csv):
    increment += 1
    save_csv = "/scratch/bf996/vlhub/logs/dcc-datacomp-results-{}.csv".format(increment)

script_path = 'python src/training/main.py --batch-size=128 --workers=8 --dc_eval --dc_eval_data_dir "/scratch/projects/hegdelab/bf996/datasets/datacomp_ds" --report-to wandb --model=RN50 --resume={} --zeroshot-frequency=1 --save-results-to-csv={};'

pythonpath_cmd = 'export PYTHONPATH="$PYTHONPATH:/scratch/bf996/vlhub/src"'

subprocess.run(pythonpath_cmd, shell=True)

for idx, model in tqdm(enumerate(model_list)):
    command = script_path.format(model, save_csv)
    try:
        subprocess.run(command, shell=True)
    except Exception as e:
        print(e)
        continue