# importiong the modules
import pandas as pd
import numpy as np
import math

headers = ['filename','angle','time','time1','width','height','frame_id','angle_convert','angle_smooth','angle_smooth_convert','brake_input','gear_state','gear_cmd','throttle_input','ax','ay','az','angle_smooth.1']
df = pd.read_csv('./chen_old/data.txt',sep="\s|,", engine='python',names=headers)
df['angle_convert_org']=df['angle'].apply(lambda x: math.radians(x))
# print(df.tail(n=10))
df.to_csv('chen_old_all.csv', index = False)

# training_data = df.sample(frac=0.7, random_state=25)
# training_data.to_csv('chen_old_training.csv', index = False)
# print(f"No. of training examples: {training_data.shape[0]}")

# testing_data = df.drop(training_data.index)
# testing_data.to_csv('chen_old_testing.csv', index = False)
# print(f"No. of testing examples: {testing_data.shape[0]}")
