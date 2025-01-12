#min_x: 26.999998312177805
#max_x: 339.0000112887177
#min_y: 99.99998030598842
#max_y: 784.0000095285399
#移除combined data 編號6的shot type資料
#預測球是在左邊右邊
import pandas as pd
import numpy as np
from collections import Counter
# 讀取六個CSV檔案
set1 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess1.csv")
set2 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess2.csv")
set3 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess3.csv")
set4 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess4.csv")
set5 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess5.csv")
set6 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess6.csv")

    

#combined_data = pd.concat([set1, set2], ignore_index=True)
combined_data = pd.concat([set1, set2,set3, set4,set5, set6], ignore_index=True)

for i in range(0,len(combined_data)):
    if combined_data['ball_round'][i]==1:
        combined_data['hit_x'][i]=combined_data.iloc[i].iloc[19+(ord(combined_data['player'][i])-ord('A'))*2]
        combined_data['hit_y'][i]=combined_data.iloc[i].iloc[20+(ord(combined_data['player'][i])-ord('A'))*2]
        
        


for i in range(0,combined_data.shape[0]):
  if(combined_data.at[i, 'ball_type']=="擋小球" or combined_data.at[i, 'ball_type']=="放小球" or combined_data.at[i, 'ball_type']=="小平球"):
    combined_data.at[i, 'ball_type'] = '網前小球'
  elif(combined_data.iloc[i]['ball_type']=="防守回挑"):
    combined_data.at[i, 'ball_type'] = '挑球'
  elif(combined_data.iloc[i]['ball_type']=="防守回抽" or combined_data.iloc[i]['ball_type']=="後場抽平球"):
    combined_data.at[i, 'ball_type'] = "平球"
    
for i in range(0,len(combined_data)):
    if combined_data['player'][i]=='A':
      combined_data['player'][i]=1
    elif(combined_data['player'][i]=='B'):
      combined_data['player'][i]=2
    elif(combined_data['player'][i]=='C'):
      combined_data['player'][i]=3
    elif(combined_data['player'][i]=='D'):
      combined_data['player'][i]=4
    if(combined_data['ball_type'][i]=='網前小球'):
     combined_data['ball_type'][i]=1
    elif(combined_data['ball_type'][i]=='挑球'):
      combined_data['ball_type'][i]=2
    elif(combined_data['ball_type'][i]=='平球'):
     combined_data['ball_type'][i]=3
    elif(combined_data['ball_type'][i]=='推撲球'):
      combined_data['ball_type'][i]=4
    elif(combined_data['ball_type'][i]=='殺球'):
      combined_data['ball_type'][i]=5
    elif(combined_data['ball_type'][i]=='切球'):
      combined_data['ball_type'][i]=6 #交換位置
    elif(combined_data['ball_type'][i]=='長球'):
      combined_data['ball_type'][i]=7#交換位置
    elif(combined_data['ball_type'][i]=='發長球'):
     combined_data['ball_type'][i]=8
    elif(combined_data['ball_type'][i]=='發短球'):
      combined_data['ball_type'][i]=9
      
# 14,15 hitx hity
combined_data = combined_data.iloc[:,[1, 2, 8, 10,13,14,16,17,18,19,20,21,22,23,24,25]] #加入return_x return_y


min_x = 26.999998312177805
max_x = 339.0000112887177
min_y = 99.99998030598842
max_y = 784.0000095285399

center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2

double_center_x=(min_x + max_x)
double_center_y=(min_y + max_y)

#combined_data = pd.concat([combined_data, symmetry_data], ignore_index=True)

# count=0
# for i in range(0,len(combined_data)//2):
#     if combined_data['return_x'][i]!=combined_data['return_x'][i+len(combined_data)//2]:
#         count+=1
        
# print(count)



X_data_list = []
count=0
for i in range(0,combined_data.shape[0]-1):
  if(combined_data['rally'][i]!=combined_data['rally'][i+1] or i==combined_data.shape[0]-2):
    X_sample_list = []
    run=i+1
    if i==combined_data.shape[0]-2:
        run=i+2
    for j in range(count,run):
      X_sample_list.append(combined_data.iloc[j].values)
    X_data_list.append(X_sample_list)
    count=i+1



 
# X = []
# Y = []
# for i in range(len(X_data_list)):
#         new_sublist = []
#         for j in range(len(X_data_list[i])):
#             if  j>=1 and X_data_list[i][j][0]!=X_data_list[i][j-1][0]:
#               break
#             if X_data_list[i][j][1] <= 9 and X_data_list[i][j][1]!=len(X_data_list[i]):
#                 new_sublist.append(X_data_list[i][j])
#         if(new_sublist!=[]):
#           X.append(new_sublist)
#           Y.append(X_data_list[i][len(new_sublist)])


X = []
Y = []
window_size = 5   # 設定滑動視窗大小

for i in range(len(X_data_list)):
    for j in range(len(X_data_list[i]) - window_size):
        window = X_data_list[i][j:j + window_size]  # 取得滑動視窗的子列表
        X.append(window)
        Y.append(X_data_list[i][j + window_size])

            
tempX=[]
tempY=[]

count=0
for i in range (0,len(Y)):
    if(Y[i][3]!=7): #第三個index是shot_type
        tempY.append(Y[i])
        tempX.append(X[i])
    else:
        count+=1
print(count)        
#print(count)
X=tempX
Y=tempY



tempX=[]
tempY=[]

count=0
for i in range (0,len(X)):
    flag=1
    for j in range(0,len(X[i])):
        if(X[i][j][3]==7):
            flag=0
    if flag==1:
        tempY.append(Y[i])
        tempX.append(X[i])

X=tempX
Y=tempY
# for i in range(len(X)):
#   len_x=len(X[i])
#   if((len_x)<9):#prepadding
#     count=9
#     temp=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] ## 因為後面會del row_list的動作，所以多加了三個0
#     for j in range(0,count-len_x):
#       X[i].insert(0,temp)
      
temp=[]
for i in range(0,len(X)):
  for j in range(0,len(X[i])):
    row_list = list(X[i][j])
    player=X[i][j][2]
    shot_type=X[i][j][3]
    del row_list[1]
    del row_list[1]
    del row_list[1]   #將ball_round刪除
    if(player==1):
      row_list[1:1] = [1,0,0,0]
    elif(player==2):
      row_list[1:1] = [0,1,0,0]
    elif(player==3):
      row_list[1:1] = [0,0,1,0]
    elif(player==4):
      row_list[1:1] = [0,0,0,1]
    if(shot_type==1):
      row_list[5:5] = [1,0,0,0,0,0,0,0]
    elif(shot_type==2):
      row_list[5:5] = [0,1,0,0,0,0,0,0]
    elif(shot_type==3):
      row_list[5:5] = [0,0,1,0,0,0,0,0]
    elif(shot_type==4):
      row_list[5:5] = [0,0,0,1,0,0,0,0]
    elif(shot_type==5):
      row_list[5:5] = [0,0,0,0,1,0,0,0]
    elif(shot_type==6): 
      row_list[5:5] = [0,0,0,0,0,1,0,0]
    elif(shot_type==8):
      row_list[5:5] = [0,0,0,0,0,0,1,0]
    elif(shot_type==9):
      row_list[5:5] = [0,0,0,0,0,0,0,1]
    temp.append(row_list)

X=temp


X=np.array(X)
Y=np.array(Y)

X=X.reshape(int(len(X)/window_size),window_size,25)

#X=X.reshape(1011,5,24)

#X=X.reshape(1087,4,24) # 如果看四回合
#X=X.reshape(940,5,24) # 如果看前五回合
#X=X.reshape(819,6,24)# 如果看六回合
#X=X.reshape(712,7,24)# 如果看七回合
X = X.astype(np.float32)
Y = Y.astype(np.float32)



import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense,Add
from keras.optimizers import Adam
from tensorflow.keras import layers
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))



for i in range(X.shape[0]):  # 遍历每个数组
  for j in range(X.shape[1]):  # 遍历数组中的每个元素
    X[i, j][13:] = scalerX.fit_transform(X[i, j][13:].reshape(-1, 1)).reshape(-1)  # 特征缩放

#----------------------------------------------------predict


from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

Y_shot_type = Y[:, 3].reshape(-1, 1)
Y_shot_type=Y_shot_type.astype(int)
Y_shot_type=Y_shot_type-1
n_classes=6
one_hot_encoded = np.zeros((Y_shot_type.shape[0], n_classes))

# 步驟3: 將對應位置設置為1
for i, val in enumerate(Y_shot_type):
    one_hot_encoded[i, val] = 1


#X_train=X[0:124,:,0:14]
#X_val=X[124:164,:,0:16]
#X_test=X[164:198,:,0:16]
# 將已經one-hot編碼的shot type標籤賦值給Y_shot_type
Y_shot_type = one_hot_encoded
# count=0
# for i in range(0,len(Y_shot_type)//2):
#     if Y_shot_type[i]!=Y_shot_type[i+len(Y_shot_type)//2]:
#         count+=1
        
# print(count)

# 將數據集分成訓練集、驗證集和測試集
#X=X[:,:,1:14]  #這行程式碼代表不考慮球員的位置和回合數


# 匯入TensorFlow和Keras相關庫
import tensorflow as tf
from tensorflow.keras import layers, Input, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train_labels), y=Y_train_labels)
# class_weights_dict = dict(enumerate(class_weights))
# max_weight = 1.3
# min_weight = 0.1
from sklearn.model_selection import KFold
#kf = KFold(n_splits=10,shuffle=True)
accs = []
history_list = []
num_classes=6 #總共有6種shot_tpye
total_cm = np.zeros((num_classes, num_classes))
val_loss_list = []
train_loss_list=[]
np.set_printoptions(suppress=True)# 設置數字顯示格式，不要有1.6e+02這種顯示方式
plt.rc('font', family='Microsoft JhengHei')

court_min_x = 26.999998312177805
court_max_x = 339.0000112887177
court_min_y = 99.99998030598842
court_max_y = 784.0000095285399

#center_x = (court_min_x + court_max_x) / 2
#center_y = (court_min_y + court_max_y) / 2
import itertools
double_center_x=(court_min_x + court_max_x)
double_center_y=(court_min_y + court_max_y)





#X=X[0:40,:,:]
num_samples = len(X)//2
from sklearn.model_selection import StratifiedKFold
num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=False)


    
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model


import numpy as np
from sklearn.model_selection import StratifiedKFold
#
#X=X[:,:,5:13]
#X=X[:,:,1:]
from sklearn.utils import shuffle


X=X[:,:,1:15]

import seaborn as sns


for train_index, val_index in skf.split(np.arange(X.shape[0]//2), np.argmax(Y_shot_type[:X.shape[0]//2], axis=1)):
    
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y_shot_type[train_index], Y_shot_type[val_index]

    
    X_train, Y_train = shuffle(X_train, Y_train) #對每個fold 做shuffle
    X_val, Y_val = shuffle(X_val, Y_val)
    # 將每個時間步展開
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_val_flattened = X_val.reshape(X_val.shape[0], -1)
    
    X_train_flattened = np.nan_to_num(X_train_flattened, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val_flattened = np.nan_to_num(X_val_flattened, nan=0.0, posinf=1e6, neginf=-1e6)
    
    #計算類別權重
    Y_train_labels = np.argmax(Y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train_labels), y=Y_train_labels)
    class_weights_dict = dict(enumerate(class_weights))
    max_weight = 1.5
    min_weight = 0.8

    adjusted_class_weights = {k: min(max_weight, max(min_weight, v)) for k, v in class_weights_dict.items()}
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # 構建決策樹模型
    model = DecisionTreeClassifier(class_weight=adjusted_class_weights)

    # 訓練模型
    model.fit(X_train_flattened, np.argmax(Y_train, axis=1))

    # 驗證模型
    y_pred = model.predict(X_val_flattened)
    accuracy = accuracy_score(np.argmax(Y_val, axis=1), y_pred)

    print(f"Validation Accuracy: {accuracy}")
    
    # 計算並印出 confusion matrix
    conf_matrix = confusion_matrix(np.argmax(Y_val, axis=1), y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    