import edge_generate as eg
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import image_to_coordinate as itc
import json

object_path = filedialog.askopenfilename( title="Select OBJ File", filetypes=(("OBJ files", "*.obj"), ("All files", "*.*")))
object_name = os.path.basename(object_path)

image_list, quaternion_list = eg.render_basic(object_path,1.15,600,800,300)
color = image_list[1]
transform = quaternion_list[1]
coordinates = []

def click_event(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    coordinates.append([x,y])
    if len(coordinates) == 2:
      cv2.destroyAllWindows()

cv2.imshow("color", color)
cv2.setMouseCallback("color", click_event)

cv2.waitKey(0)

if len(coordinates) == 2:
  x_direction = [coordinates[1][0] - coordinates[0][0], coordinates[1][1] - coordinates[0][1]]
  print("Coordinates:", coordinates[0])
  print("X Direction:", x_direction)
  transform_combined = itc.main(object_path,transform,coordinates[0],coordinates[1],800,600,300,1.15)
  if transform_combined is None:
    print("QR code not on object.")
    messagebox.showerror("Error", "QR code is placed outside the object.")

  # kanamono_typeの変数
  kanamono_type = object_name

  # JSONフォーマット用の辞書に変換
  data = {
      "kanamono_type": kanamono_type,
      "position": transform_combined['translation'].tolist(),  # numpy配列をリストに変換
      "rotation": transform_combined['quaternion'].tolist()    # numpy配列をリストに変換
  }

  data2 = {
      "kanamono_type": kanamono_type,
      "position": transform['translation'].tolist(),  # numpy配列をリストに変換
      "rotation": transform['quaternion'].tolist()    # numpy配列をリストに
  }

  output_dir = filedialog.askdirectory(title="Select Output Directory")
  output_path = os.path.join(output_dir, "test.json")
  output_path2 = os.path.join(output_dir, "test2.json")

  with open(output_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)
  
  with open(output_path2, 'w') as json_file:
    json.dump(data2, json_file, indent=4)

else:
  print("Two points were not selected.")



def read_json(json_path):
  with open(json_path, 'r') as json_file:
    data = json.load(json_file)
  return data