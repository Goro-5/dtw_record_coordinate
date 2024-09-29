import edge_generate as eg
import os
import tkinter as tk
from tkinter import filedialog
import json

root = tk.Tk()
root.withdraw()  # Hide the root window

file_path = filedialog.askopenfilename(
  title="Select OBJ File",
  filetypes=(("OBJ files", "*.obj"), ("All files", "*.*"))
)

#　objファイルパスのファイル名部分を取得
file_name = os.path.basename(file_path)
# ファイル名の拡張子を削除
obj_name = os.path.splitext(file_name)[0]

quaternion = eg.applied_transform_quaternion2(eg.load_obj(file_path),"xm")
output_dir = filedialog.askdirectory(title="Select Output Directory")
output_path = os.path.join(output_dir, "test.json")

# kanamono_typeの変数
kanamono_type = obj_name

# JSONフォーマット用の辞書に変換
data = {
    "kanamono_type": kanamono_type,
    "position": quaternion['translation'].tolist(),  # numpy配列をリストに変換
    "rotation": quaternion['quaternion'].tolist()    # numpy配列をリストに変換
}

with open(output_path, 'w') as json_file:
  json.dump(data, json_file, indent=4)


def read_json(json_path):
  with open(json_path, 'r') as json_file:
    data = json.load(json_file)
  return data