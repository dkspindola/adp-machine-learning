import json 
from pandas import DataFrame
import os
import pandas as pd

def convert_to_csv(json_file: dict[str]):
    d = json.load(open(json_file, "r"))
    df = DataFrame.from_dict(d, orient='index')
    df.transpose
    df.to_csv(os.path.splitext(json_file)[0] + '.csv')

convert_to_csv('build/validate_average/cnn-corvin-30-trials/1738768485/averages.json')
convert_to_csv('build/validate_average/cnn-corvin-60-trials/1738768542/averages.json')
convert_to_csv('build/validate_average/cnn-davi/1738768834/averages.json')
convert_to_csv('build/validate_average/cnn-davi-new/1738768586/averages.json')


