# importiong the modules
import pandas as pd
import numpy as np
import math

headers = [
    "filename",
    "angle",
    "time",
    "time1",
    "width",
    "height",
    "frame_id",
    "angle_convert",
    "angle_smooth",
    "angle_smooth_convert",
    "brake_input",
    "gear_state",
    "gear_cmd",
    "throttle_input",
    "ax",
    "ay",
    "az",
    "angle_smooth.1",
]
df = pd.read_csv("./data.txt", sep="\s|,", engine="python", names=headers)
df["angle_convert_org"] = df["angle"].apply(lambda x: math.radians(x))
# print(df.tail(n=10))
df.to_csv("chen_old_all.csv", index=False)
