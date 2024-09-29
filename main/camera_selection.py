import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def on_image_click(event, index):
    global selected_camera_index
    selected_camera_index = index
    for label in camera_labels:
        label.config(borderwidth=0)
    event.widget.config(borderwidth=2, relief="solid")

def show_camera_images():
    for i, index in enumerate(available_cameras):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Camera index {index} out of range")
            continue
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img.thumbnail((160, 120))
            imgtk = ImageTk.PhotoImage(image=img)

            label = tk.Label(frame_container, image=imgtk, borderwidth=0)
            label.image = imgtk  # 参照を保持する必要がある
            label.grid(row=0, column=i, padx=10, pady=10)
            
            label.bind("<Button-1>", lambda event, idx=index: on_image_click(event, idx))

            camera_labels.append(label)

        cap.release()

def find_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def on_confirm():
    global root
    root.quit()

def select_camera():
    global root, frame_container, available_cameras, camera_labels, selected_camera_index
    root = tk.Tk()
    root.title("Camera Selector")

    available_cameras = find_available_cameras()
    selected_camera_index = None
    camera_labels = []

    frame_container = tk.Frame(root)
    frame_container.pack()

    show_camera_images()

    confirm_button = tk.Button(root, text="決定", command=on_confirm)
    confirm_button.pack(pady=10)

    root.mainloop()

    return selected_camera_index

if __name__ == '__main__':
  # 他のプログラムから呼び出す
  camera_number = select_camera()
  cv2.waitKey(0)
  if camera_number is not None:
      print(f"選択されたカメラ番号: {camera_number}")
  else:
      print("利用可能なカメラがありません")
