# ================== IMPORT PACKLAGES 

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg

# ============= READ INPUT VIDEO ================

# Open the video file.
filename = askopenfilename()
cap = cv2.VideoCapture(filename)
Frames_all = []
# Loop over the frames in the video.
while True:
    # Read the next frame from the video.
    ret, frame = cap.read()

    # If the frame is not read successfully, break from the loop.
    if not ret:
        break

    # Convert the frame to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the frame.
    cv2.imshow('Frame', frame)
    Frames_all.append(frame)
    # Wait for a key press.
    key = cv2.waitKey(1)

    # If the key pressed is `q`, break from the loop.
    if key == ord('q'):
        break

# Close the video file.
cap.release()

# Destroy all windows created by OpenCV.
cv2.destroyAllWindows()


# ===================== CONVERT VIDO INTO FRAMES ===================


Testfeature = []

for iiij in range(0,len(Frames_all)):
    
    img1 = Frames_all[iiij]
    
    plt.imshow(img1)
    plt.title('ORIGINAL IMAGE')
    plt.show()
    
    #
    # PRE-PROCESSING
    
    h1=512
    w1=512
    
    dimension = (w1, h1) 
    resized_image1 = cv2.resize(img1,(h1,w1))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image1)
    plt.show()
    
    
    # ===== FEATURE EXTRACTION ====
    
    
    #=== MEAN STD DEVIATION ===
    
    mean_val = np.mean(resized_image1)
    median_val = np.median(resized_image1)
    var_val = np.var(resized_image1)
    features_extraction = [mean_val,median_val,var_val]
    
    print("====================================")
    print("        Feature Extraction          ")
    print("====================================")
    print()
    print(features_extraction)    
    
    
    # ==== LBP =========
    
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
       
          
    def find_pixel(imgg, center, x, y):
        new_value = 0
        try:
            if imgg[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value
       
    # Function for calculating LBP
    def lbp_calculated_pixel(imgg, x, y):
        center = imgg[x][y]
        val_ar = []
        val_ar.append(find_pixel(imgg, center, x-1, y-1))
        val_ar.append(find_pixel(imgg, center, x-1, y))
        val_ar.append(find_pixel(imgg, center, x-1, y + 1))
        val_ar.append(find_pixel(imgg, center, x, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y))
        val_ar.append(find_pixel(imgg, center, x + 1, y-1))
        val_ar.append(find_pixel(imgg, center, x, y-1))
        power_value = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_value[i]
        return val
       
       
    height, width, _ = img1.shape
       
    img_gray_conv = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
       
    img_lbp = np.zeros((height, width),np.uint8)
       
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)
    
    plt.imshow(img_lbp, cmap ="gray")
    plt.title("LBP")
    plt.show()   


#============================ 5. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split

test= os.listdir('Dataset/Test')
train = os.listdir('Dataset/Train')

#       
dot1= []
labels1 = [] 
for img11 in test:
        # print(img)
        img_1 = mpimg.imread('Dataset/Test//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)


for img11 in train:
        # print(img)
        img_1 = mpimg.imread('Dataset/Train//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)


x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of test data   :",len(x_train))
print("Total no of train data  :",len(x_test))


#============================ CLASSIFICATION =================================

# === DECISION TREE ===
    
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier() 


from keras.utils import to_categorical

x_train1=np.zeros((len(x_train),50))
for i in range(0,len(x_train)):
        x_train1[i,:]=np.mean(x_train[i])

x_test1=np.zeros((len(x_test),50))
for i in range(0,len(x_test)):
        x_test1[i,:]=np.mean(x_test[i])


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)



clf.fit(x_train1,y_train1)


y_pred = clf.predict(x_test1)

y_pred_tr = clf.predict(x_train1)


from sklearn import metrics

acc_dt=metrics.accuracy_score(y_pred_tr,y_train1)*100

print("-----------------------------------------------")
print("Machine Learning ---- > Decision Tree          ")
print("-----------------------------------------------")
print()
print("1. Decision Tree Accuracy  = ",acc_dt,'%' )
print()
print("2.Classification Report ")
print()
print(metrics.classification_report(y_pred_tr,y_train1))


# ============= RF ================

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(x_train1,y_train1)

y_pred_rf = rf.predict(x_test1)

y_pred_tr = rf.predict(x_train1)

acc_rf=metrics.accuracy_score(y_pred_tr,y_train1)*100

print("-----------------------------------------------")
print("Machine Learning ---- > Random Forest     ")
print("-----------------------------------------------")
print()
print("1. Random Forest = ",acc_rf,'%' )
print()
print("2.Classification Report ")
print()
print(metrics.classification_report(y_pred_tr,y_train1))



# ======================= PREDICTION 


# Initialize variables
video_path = filename
output_video_path = 'output_video.mp4'
json_file_path = '1.json'

# Load the JSON file containing bounding box information
with open(json_file_path, 'r') as json_file:
    bounding_boxes = json.load(json_file)

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Video file not found.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Draw bounding boxes if the frame has associated bounding box data
    for box_data in bounding_boxes:
        if box_data['frame'] == frame_count:
            x, y, width, height = box_data['bounding_box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green bounding box

    out.write(frame)  # Write the frame with bounding boxes to the output video

# Release video capture and video writer
cap.release()
out.release()

# print(f"Bounding boxes added to the video and saved to {output_video_path}.")
