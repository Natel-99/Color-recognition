from os import listdir
import cv2
import numpy as np
import pickle
import os

raw_folder = "a"
dest_size = (64, 64)

print("Bắt đầu xử lý ảnh...")

pixels = []
labels = []

for folder in listdir(raw_folder):
    if folder[0] != ".":  # Bỏ qua cac thư mục không hợp lệ
        print("Processing folder ", folder)
        for file in listdir(os.path.join(raw_folder,folder)):
            if file[0] != '.':
                print("---- Processing file = ", file)
                pixels.append(cv2.resize(cv2.imread(os.path.join(raw_folder, folder, file)), dsize=dest_size))
                labels.append(folder)

pixels = np.array(pixels)
labels = np.array(labels)


from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
print(encoder.classes_)

file = open('dataA.pkl', 'wb')
pickle.dump((pixels, labels), file)
file.close()
#pixel lưu 1 list các ảnh
#label là nhãn của ảnh

file = open('labelsA.pkl', 'wb')
pickle.dump(encoder, file)
file.close()
#Không cần load từng folder chỉ cần load từ data.pkl,labels.pkl