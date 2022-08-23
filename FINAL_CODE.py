import pickle
from keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
import csv
import time

tf_session = tf.compat.v1.Session()
tf_graph = tf.compat.v1.get_default_graph()
with tf_session.as_default():
    with tf_graph.as_default():
        model = load_model("model_categorical_complex11h36.h5")
        model.load_weights("color_weights11h36.hdf5")
file = open('labels11h36.pkl', 'rb')
encoder = pickle.load(file)
file.close()

def change_brightness(img, value):
# Điều chỉnh độ sáng của ảnh thu được bởi thiết bị thu hình ảnh
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def adjust_image_gamma(image, gamma):
# Điều chỉnh độ tương phản của ảnh thu được bởi thiết bị thu hình ảnh
  image = np.power(image, gamma)
  max_val = np.max(image.ravel())
  image = image/max_val * 255
  image = image.astype(np.uint8)
  return image
def strings_to_number(x):
# Đặt số tương ứng với mỗi màu, excel tránh được các lỗi liên quan chuỗi kí tự
    return  {'White': 1, 'Black': 2, 'Red': 3,
            'Orange': 4, 'Yellow': 5, 'Blue': 6,
            'Green': 7, 'Brown': 8, 'Violet': 9}[x]
def xla_model(a):
        frame = a
        frame = cv2.resize(frame, dsize=(64, 64))
# mở rộng chiều
        frame = np.expand_dims(frame, axis=0)
        with tf_session.as_default():
            with tf_graph.as_default():
                result = model.predict(frame)[0]
                #print(result)
                if len(result)>1:
                    result_id = np.argmax(result)
                    if result[result_id] > tile:
                        ID = encoder.classes_[result_id]
                    else:
                        ID = "Unknown"
        return ID, result[result_id]

# Webcam Laptop
cap = cv2.VideoCapture(0)
# Camera thiết bị qua Mạng Lan
# cap = cv2.VideoCapture('http://192.168.219.66:8080/video')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Đánh dấu lại thời gian C là thời gian bắt đầu chạy
B = time.time()
C = B
# Tạo các giá trị ban đầu cho các biến
tile = 0.8
Chu = " "
K = " "
D = True
while cap.isOpened():
# Thời gian A gọi liên tục trong vòng lặp while
# --> A-C là thời gian chương trình đã chạy
    A = time.time()
    diff = cv2.absdiff(frame1, frame2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    cv2.imshow('AAA', dilated)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (A > B):
# Sau t giây, bật chức năng lưu dữ liệu vào excel 1 lần
        B+=4
        D=True
    for contour in contours:
# Tìm ra Bounding box hình chữ nhật đứng
        (x, y, w, h) = cv2.boundingRect(contour)
        if (300 > w > 100 and 300 > h > 100):
# Tách khung hình ra, xử lý ảnh Resnet
            letter = frame1[y + h//4:y + 3*h//4, x + w//4:x + 3*w//4]
            # letter=cv2.resize(letter,(480,270))
            letter1 = change_brightness(letter, value=-30)
            letter1 = adjust_image_gamma(letter1, gamma = 2.0)
            letter2=cv2.resize(letter1,(480,500))
            #cv2.imshow('letter1',letter2)
            ID1, ID2 = xla_model(letter1)
# Vẽ khung, điền chữ
            if (ID1 != "Unknown"):
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame1, ID1+' : '+str(ID2), (x, y - 10), 0, 1, (0, 255, 0), 2)
                Chu = ID1
# Lưu dữ liệu vào excel
#                 if (D == True):
#                     print(Chu)
#                     K = strings_to_number(Chu)
#                     print(round(A-C, 2))
#                     with open(r"C:\Users\phatn\OneDrive\Desktop\20202_AI_In_Robot\simulink\names.csv", "a+", newline='') as file:
#                         fieldnames = ['Thoi_gian', 'Mau', 'So_hoa']
#                         writer = csv.DictWriter(file, fieldnames=fieldnames)
#                         writer.writerow({'Thoi_gian': round(A-C, 2), 'Mau': Chu, 'So_hoa': K})
#                     with open(r"C:\Users\phatn\OneDrive\Desktop\20202_AI_In_Robot\simulink\ThoiGian.csv", "a+", newline='') as file:
#                         writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                         writer.writerow({round(A-C, 2)})
#                     with open(r"C:\Users\phatn\OneDrive\Desktop\20202_AI_In_Robot\simulink\MauSac.csv", "a+", newline='') as file:
#                         writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                         writer.writerow({K})
#                     D = False
    #frame1=cv2.resize(frame1,(480,270))
    cv2.imshow("Video", frame1)
    #abc=change_brightness(frame1, value=-30)
    #cv2.imshow("Video_anhsang", abc)
    #cde = adjust_image_gamma(frame1, gamma = 2.0)
    #cv2.imshow("Video_dotuongphan", cde)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(50) == 27:
        break
cap.release()
cv2.destroyAllWindows()
