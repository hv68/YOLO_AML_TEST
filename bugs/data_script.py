import pandas as pd 
import numpy as np
import os.path
import cv2
import math

PBW = 0
ABW = 1

# creating training and validation set
df = pd.read_csv("Train_bugs.csv")

df = df[df['image_id_worm'].apply(lambda s: os.path.exists(f"./images/{s}"))]

df_train_txt = df.iloc[:df.shape[0]*70//100]
df_val_txt = df.iloc[df.shape[0]*70//100:]

df_train_txt["image_id_worm"] = df_train_txt["image_id_worm"].apply(lambda x: "../datasets/bugs/images/"+x)
df_val_txt["image_id_worm"] = df_val_txt["image_id_worm"].apply(lambda x: "../datasets/bugs/images/"+x)

df_train_txt = df_train_txt['image_id_worm']
df_val_txt = df_val_txt["image_id_worm"]
df_train_txt.to_csv("train.txt", sep=' ', index=False, header=False)

df_val_txt.to_csv("val.txt", sep=' ', index=False, header=False, )

# create labels for bounding boxes
df = df['image_id_worm']
bb = pd.read_csv("images_bboxes.csv")
bb = bb[bb["image_id"].isin(df.tolist())]

for i in range(df.shape[0]):
	filtered = bb[bb["image_id"] ==  df.iloc[i]]
	if filtered.shape[0] == 1 and not type(filtered.iloc[0]["worm_type"]) == str and  math.isnan(filtered.iloc[0]["worm_type"]):
		continue
	label = []
	center_x = []
	center_y = []
	width = []
	height = []
	image = cv2.imread(f'images/{df.iloc[i]}')
	for j in range(filtered.shape[0]):
		if filtered.iloc[j]['worm_type'] == "pbw":
			label.append(PBW)
		else:
			label.append(ABW)
		x_vals = []
		y_vals = []
		coor = filtered.iloc[j]['geometry'][11:-2].split(",")
		for k in coor:
			leng = 0
			if len(k.split(" ")) == 3:
				leng = 1
			x_vals.append(float(k.split(" ")[0+leng]))
			y_vals.append(float(k.split(" ")[1+leng]))
		c_x = ((max(x_vals) + min(x_vals))/2)/image.shape[0]
		c_y = ((max(y_vals) + min(y_vals))/2)/image.shape[1]
		w = (max(x_vals) - min(x_vals))/image.shape[1]
		h = (max(y_vals) - min(y_vals))/image.shape[0]
		center_x.append(c_x)
		center_y.append(c_y)
		height.append(h)
		width.append(w)
	label_txt = pd.DataFrame(data={"label": label, "x_center": center_x, "y_center": center_y, "width": width, "height": height})
	label_txt.to_csv(f"labels/{df.iloc[i].replace('jpg','txt')}", sep=' ', index=False, header=False )
