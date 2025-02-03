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

script_path = 'python src/training/main.py --batch-size=128 --workers=8 --openimages-val "/scratch/bf996/vlhub/metadata/oi1000_val.csv" --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/projects/hegdelab/bf996/datasets" --imagenet-s "/imagenet-sketch" --imagenet-a "/imagenet-a" --imagenet-r "/imagenet-r" --report-to wandb --model=RN50 --resume={} --caption-subset="in100" --extended-metrics=True --zeroshot-frequency=1;'

pythonpath_cmd = 'export PYTHONPATH="$PYTHONPATH:/scratch/bf996/vlhub/src"'

subprocess.run(pythonpath_cmd, shell=True)

for idx, model in tqdm(enumerate(model_list)):
    command = script_path.format(model)
    try:
        subprocess.run(command, shell=True)
    except Exception as e:
        print(e)
        continue