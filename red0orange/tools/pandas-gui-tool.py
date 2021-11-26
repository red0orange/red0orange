import os
import argparse
import pandas as pd
from pandasgui import show

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='pandas df file path')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    df = pd.read_excel(opt.data_path)
    show(df)
    pass