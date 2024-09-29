import cv2
import numpy as np

def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot detect camera")
        exit()

    # 解像度を設定
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 幅を1920ピクセルに設定
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1680) # 高さを1680ピクセルに設定
    n=0
    areas = []
    while True:
        ret, frame = capture.read()
        if ret:

            qr = cv2.QRCodeDetector()
            # QRコードを検出
            data, points, straight_qrcode = qr.detectAndDecode(frame)
            if points is not None and len(points) > 0:
                points = points.astype(np.int32)
                # QRコードを検出した位置に四角形を描画
                cv2.polylines(frame, [points], True, (255, 0, 0), thickness=5)
                # 四角形の面積を計算
                area = cv2.contourArea(points)
                areas.append(area)
                
                n += 1
                if n > 10:
                    
                    break

            cv2.imshow('frame',frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    area_med = np.median(areas)
    print(f"QR Code Target Area: {area_med}")
    return area_med

if __name__ == '__main__':
    target = main()
    print("done")