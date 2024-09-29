import cv2

def get_camera_info(camera_number, width):
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(camera_number)
    default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = default_width / default_height
    height = int(width / aspect_ratio)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def update_width(val):
    # 解像度を設定し、その結果が適用されているか確認
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, val)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested width: {val}, Actual width: {actual_width}\nRequested height: {val / (actual_width / actual_height)}, Actual height: {actual_height}")


def main():
    camera_number = 1  # カメラ番号を1に変更してみる
    initial_width = 5000  # 初期の幅

    global cap
    cap = None
    cap = get_camera_info(camera_number, initial_width)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # ウィンドウを作成
    cv2.namedWindow('Camera Feed')

    # Trackbarを追加
    cv2.createTrackbar('Width', 'Camera Feed', initial_width, 7000, update_width)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Camera Feed', frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
