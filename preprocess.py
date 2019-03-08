# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

final_title = "uid \t user_city \t item_id \t author_id \t item_city \t channel \t finish \t like \t music_id \t device \t creat_time \t video_duration"
final_title = final_title.replace(' ', '').split('\t')
final = pd.read_table('./dataset/final_track2_train.txt', header=None)
final.columns = final_title

#