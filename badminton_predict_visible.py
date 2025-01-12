# 注意 slice被改成了smash!!

#預測球是在左邊右邊
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 讀取六個CSV檔案
set1 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess1.csv")
set2 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess2.csv")
set3 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess3.csv")
set4 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess4.csv")
set5 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess5.csv")
set6 = pd.read_csv(r"C:\Users\chester\Desktop\badminton\preprocess6.csv")
#print(max(set5['ball_round'])) #39
#print(max(set6['ball_round'])) #27

combined_data = pd.concat([set1, set2, set3, set4, set5, set6], ignore_index=True)
for i in range(0,combined_data.shape[0]):
  if(combined_data.at[i, 'ball_type']=="擋小球" or combined_data.at[i, 'ball_type']=="放小球" or combined_data.at[i, 'ball_type']=="小平球"):
    combined_data.at[i, 'ball_type'] = '網前小球'
  elif(combined_data.iloc[i]['ball_type']=="挑球" or combined_data.iloc[i]['ball_type']=="防守回挑"):
    combined_data.at[i, 'ball_type'] = '挑球'
  elif(combined_data.iloc[i]['ball_type']=="防守回抽" or combined_data.iloc[i]['ball_type']=="平球" or combined_data.iloc[i]['ball_type']=="後場抽平球"):
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
    elif(combined_data['ball_type'][i]=='長球'):
      combined_data['ball_type'][i]=6
    elif(combined_data['ball_type'][i]=='切球'):
      combined_data['ball_type'][i]=7
    elif(combined_data['ball_type'][i]=='發長球'):
     combined_data['ball_type'][i]=8
    elif(combined_data['ball_type'][i]=='發短球'):
      combined_data['ball_type'][i]=9
      
combined_data = combined_data.iloc[:,[1, 2, 8, 10,16,17,18,19,20,21,22,23,24,25]] #加入return_x return_y


#combined_data

X_data_list = []
count=0
for i in range(0,combined_data.shape[0]-1):
  if(combined_data['rally'][i]!=combined_data['rally'][i+1]):
    X_sample_list = []
    for j in range(count,i+1):
      X_sample_list.append(combined_data.iloc[j].values)
    X_data_list.append(X_sample_list)
    count=i+1
    
#上面的目的是想要將每個rally分離


X = []
Y = []
window_size = 5  # 設定滑動視窗大小

for i in range(len(X_data_list)):
    for j in range(len(X_data_list[i]) - window_size):
        window = X_data_list[i][j:j + window_size]  # 取得滑動視窗的子列表
        
        # 檢查視窗內的元素是否符合條件
        valid_window = True
        for k in range(1, window_size):
            if window[k][0] != window[k-1][0]:
                valid_window = False
                break
        
        # 如果符合條件，將視窗加入X，並將視窗之後的一個元素加入Y
        if valid_window:
            X.append(window)
            Y.append(X_data_list[i][j + window_size])
          
      
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
      row_list[5:5] = [1,0,0,0,0,0,0,0,0]
    elif(shot_type==2):
      row_list[5:5] = [0,1,0,0,0,0,0,0,0]
    elif(shot_type==3):
      row_list[5:5] = [0,0,1,0,0,0,0,0,0]
    elif(shot_type==4):
      row_list[5:5] = [0,0,0,1,0,0,0,0,0]
    elif(shot_type==5):
      row_list[5:5] = [0,0,0,0,1,0,0,0,0]
    elif(shot_type==6):
      row_list[5:5] = [0,0,0,0,0,1,0,0,0]
    elif(shot_type==7):
      row_list[5:5] = [0,0,0,0,0,0,1,0,0]
    elif(shot_type==8):
      row_list[5:5] = [0,0,0,0,0,0,0,1,0]
    elif(shot_type==9):
      row_list[5:5] = [0,0,0,0,0,0,0,0,1]
    temp.append(row_list)

X=temp




Y=np.array(Y)




    
    
X=np.array(X)
X=X.reshape(int(X.shape[0]/5),5,24)

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
    X[i, j][14:24] = scalerX.fit_transform(X[i, j][14:24].reshape(-1, 1)).reshape(-1)  # 特征缩放

#----------------------------------------------------predict






# 匯入TensorFlow和Keras相關庫
import tensorflow as tf
from tensorflow.keras import layers, Input, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

Y_shot_type = Y[:, 3].reshape(-1, 1)
Y_shot_type=Y_shot_type-1
Y_shot_type=Y_shot_type.astype(int)
n_classes=7
one_hot_encoded = np.zeros((Y_shot_type.shape[0], n_classes))

# 步驟3: 將對應位置設置為1
for i, val in enumerate(Y_shot_type):
    one_hot_encoded[i, val] = 1
Y_shot_type = one_hot_encoded
    
X_train = X[0:740] #前四場當訓練集

X_val = X[740:890] #第五場當驗證集
X_test = X[890:] #第六場當測試集

# 將目標標籤同樣分成訓練集、驗證集和測試集
Y_train = Y_shot_type[0:740]
Y_val = Y_shot_type[740:890]
Y_test = Y_shot_type[890:]





# 定義模型的輸入層
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
input_shot_type=input_layer[:,:,5:14]
# 嵌入層
x = layers.Dense(128, activation='relu')(input_shot_type)

# 多頭注意力層和殘差連接
# attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
# #attention_output = layers.Attention()([x, x])
# x = layers.Add()([x, attention_output])  # 殘差連接
# x = layers.LayerNormalization()(x)  # 添加規範層

# LSTM 層
x = layers.LSTM(units=128, return_sequences=False)(x)
#x = layers.Dropout(0.1)(x)

# 全連接層
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

# 定義輸出層，使用softmax激活函數
output_layer = layers.Dense(Y_train.shape[1], activation="softmax")(x)

# 創建並編譯模型
model_shot_type = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model_shot_type.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 定義回調函數
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 訓練模型
history_shot_type=model_shot_type.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32, callbacks=[reduce_lr, early_stopping])#
#model_shot_type.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32)

## 預測測試集
model_shot_type_predict = model_shot_type.predict(X_test)
model_shot_type_predict = np.argmax(model_shot_type_predict, axis=1)  # 將預測結果轉為類別索引
original_Y_test_labels = np.argmax(Y_test, axis=1)  # 將真實標籤轉為類別索引

# 計算準確率
acc = 0
for i in range(len(original_Y_test_labels)):
    if original_Y_test_labels[i] == model_shot_type_predict[i]:
        acc += 1
print("acc:", acc / len(original_Y_test_labels))

model_shot_type_predict=model_shot_type_predict+1






Y_next_ball=Y[:,4:6]
scalerY = MinMaxScaler(feature_range=(0, 1))
Y_next_ball = scalerY.fit_transform(Y_next_ball)



X_train = X[0:740] #前四場當訓練集

X_val = X[740:890] #第五場當驗證集
X_test = X[890:] #第六場當測試集

# 將目標標籤同樣分成訓練集、驗證集和測試集
Y_train = Y_next_ball[0:740]
Y_val = Y_next_ball[740:890]
Y_test = Y_next_ball[890:]

# 改进的模型架构
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.Dense(128, activation='relu')(input_layer)
attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
x = layers.Add()([x, attention_output])  # 殘差連接
x = layers.LayerNormalization()(x)  # 添加規範層

x = (layers.LSTM(units=128, return_sequences=False))(x)
x = layers.Dropout(0.3)(x)

# 全連接層
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

# 定義輸出層，使用softmax激活函數
output_layer = layers.Dense(Y_train.shape[1], activation="linear")(x)
 # 創建並編譯模型
model_next_ball = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model_next_ball.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

 # 定義回調函數
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

 # 訓練模型
history_next_ball=model_next_ball.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32, callbacks=[reduce_lr, early_stopping])

predictions_next_ball = model_next_ball.predict(X_test)

predictions_next_ball = scalerY.inverse_transform(predictions_next_ball)
Y_test= scalerY.inverse_transform(Y_test)

next_ball_MAE_LOSS_X=0
next_ball_MAE_LOSS_Y=0

next_ball_MSE_LOSS_X=0
next_ball_MSE_LOSS_Y=0

for i in range(0,len(predictions_next_ball)):
    next_ball_MAE_LOSS_X=next_ball_MAE_LOSS_X+abs(predictions_next_ball[i][0]-Y_test[i][0])
    next_ball_MAE_LOSS_Y=next_ball_MAE_LOSS_Y+abs(predictions_next_ball[i][1]-Y_test[i][1])
    
    next_ball_MSE_LOSS_X+=(predictions_next_ball[i][0]-Y_test[i][0])**2
    next_ball_MSE_LOSS_Y+=(predictions_next_ball[i][1]-Y_test[i][1])**2

next_ball_MAE_LOSS_X=next_ball_MAE_LOSS_X/len(predictions_next_ball)
next_ball_MAE_LOSS_Y=next_ball_MAE_LOSS_Y/len(predictions_next_ball)

next_ball_MSE_LOSS_X=next_ball_MSE_LOSS_X/len(predictions_next_ball)
next_ball_MSE_LOSS_Y=next_ball_MSE_LOSS_Y/len(predictions_next_ball)

plt.plot(history_next_ball.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_next_ball.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_next_ball')


# court_min_x = 26.999998312177805
# court_max_x = 339.0000112887177
# court_min_y = 99.99998030598842
# court_max_y = 784.0000095285399
# double_center_x=(court_min_x + court_max_x)
# double_center_y=(court_min_y + court_max_y)

# for i in range(0,len(predictions_next_ball)):
#     TEMPX=0
#     TEMPY=0
#     if (predictions_next_ball[i][1]<442 and predictions_next_ball[i][0]>=court_min_x and predictions_next_ball[i][0]<105) or (predictions_next_ball[i][1]>442 and predictions_next_ball[i][0]<339 and predictions_next_ball[i][0]>=261):
#         TEMPX=0
#         #count+=1
#     elif (predictions_next_ball[i][1]<442 and predictions_next_ball[i][0]>=105 and predictions_next_ball[i][0]<261) or (predictions_next_ball[i][1]>442 and predictions_next_ball[i][0]<double_center_x-105 and predictions_next_ball[i][0]>=double_center_x-261):
#         TEMPX=1
#         #count+=1
#     elif (predictions_next_ball[i][1]<442 and predictions_next_ball[i][0]>=261 and predictions_next_ball[i][0]<339) or (predictions_next_ball[i][1]>442 and predictions_next_ball[i][0]<double_center_x-261 and predictions_next_ball[i][0]>=double_center_x-339):
#         TEMPX=2
#         #count+=1
#     else:
#         TEMPX=3 # 出界
#     # elif predictions_next_ball[i][1]<442 and  predictions_next_ball[i][0]<=court_min_x:
#     #     TEMPX=3 # 出界
#     #     #count+=1
#     # else:
#     #     TEMPX=4 # 出界
#     #    # count+=1
    
    
#     if (predictions_next_ball[i][1]>=court_min_y and predictions_next_ball[i][1]<154) or (predictions_next_ball[i][1]<double_center_y-court_min_y and predictions_next_ball[i][1]>=double_center_y-154):
#          TEMPY=0
#         # count+=1
#     elif (predictions_next_ball[i][1]>=154 and predictions_next_ball[i][1]<322) or (predictions_next_ball[i][1]<double_center_y-154 and predictions_next_ball[i][1]>=double_center_y-322):
#          TEMPY=1
#          #count+=1
#     elif (predictions_next_ball[i][1]>=322 and predictions_next_ball[i][1]<442) or (predictions_next_ball[i][1]<double_center_y-322 and predictions_next_ball[i][1]>=double_center_y-442):
#          TEMPY=2
#         # count+=1
#     else:
#         TEMPY=3 # 出界
#         #count+=1   
#     predictions_next_ball[i]=[TEMPX,TEMPY]
    
# for i in range(0,len(Y_test)):
#     TEMPX=0
#     TEMPY=0
#     if (Y_test[i][1]<442 and Y_test[i][0]>=court_min_x and Y_test[i][0]<105) or (Y_test[i][1]>442 and Y_test[i][0]<339 and Y_test[i][0]>=261):
#         TEMPX=0
#         #count+=1
#     elif (Y_test[i][1]<442 and Y_test[i][0]>=105 and Y_test[i][0]<261) or (Y_test[i][1]>442 and Y_test[i][0]<double_center_x-105 and Y_test[i][0]>=double_center_x-261):
#         TEMPX=1
#         #count+=1
#     elif (Y_test[i][1]<442 and Y_test[i][0]>=261 and Y_test[i][0]<339) or (Y_test[i][1]>442 and Y_test[i][0]<double_center_x-261 and Y_test[i][0]>=double_center_x-339):
#         TEMPX=2
#         #count+=1
#     else:
#         TEMPX=3 # 出界
#     # elif Y_test[i][1]<442 and  Y_test[i][0]<=court_min_x:
#     #     TEMPX=3 # 出界
#     #     #count+=1
#     # else:
#     #     TEMPX=4 # 出界
#     #    # count+=1
    
    
#     if (Y_test[i][1]>=court_min_y and Y_test[i][1]<154) or (Y_test[i][1]<double_center_y-court_min_y and Y_test[i][1]>=double_center_y-154):
#          TEMPY=0
#         # count+=1
#     elif (Y_test[i][1]>=154 and Y_test[i][1]<322) or (Y_test[i][1]<double_center_y-154 and Y_test[i][1]>=double_center_y-322):
#          TEMPY=1
#          #count+=1
#     elif (Y_test[i][1]>=322 and Y_test[i][1]<442) or (Y_test[i][1]<double_center_y-322 and Y_test[i][1]>=double_center_y-442):
#          TEMPY=2
#         # count+=1
#     else:
#         TEMPY=3 # 出界
#         #count+=1   
#     Y_test[i]=[TEMPX,TEMPY]
    
# P_X=predictions_next_ball[:,0]
# P_Y=predictions_next_ball[:,1]

# TEST_X=Y_test[:,0]
# TEST_Y=Y_test[:,1]
# count_X=0
# for i in range(0,len(P_X)):
#     if P_X[i]==TEST_X[i]:
#         count_X+=1
# count_Y=0
# for i in range(0,len(P_Y)):
#     if P_Y[i]==TEST_Y[i]:
#         count_Y+=1
        
# print("X預測準確率:",count_X/len(P_X))
# print("Y預測準確率:",count_Y/len(P_Y))

# cm = confusion_matrix(P_X, TEST_X, labels=range(4))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# #plt.xticks(rotation=45)  #X軸旋轉45度
# plt.title("predicted_A_X")
# plt.show()

# cm = confusion_matrix(P_Y, TEST_Y, labels=range(4))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# #plt.xticks(rotation=45)  #X軸旋轉45度
# plt.title("predicted_A_Y")
# plt.show()






Y_67=Y[:,6:8]
scalerY = MinMaxScaler(feature_range=(0, 1))
Y_67 = scalerY.fit_transform(Y_67)
#X_train=X[0:124,:,0:14]
#X_val=X[124:164,:,0:16]
#X_test=X[164:198,:,0:16]
X_train = X[0:740] #前四場當訓練集

X_val = X[740:890] #第五場當驗證集
X_test = X[890:] #第六場當測試集

# 將目標標籤同樣分成訓練集、驗證集和測試集
Y_train = Y_67[0:740]
Y_val = Y_67[740:890]
Y_test = Y_67[890:]
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# 改进的模型架构
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.Dense(128, activation='relu')(input_layer)
attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
x = layers.Add()([x, attention_output])  # 殘差連接
x = layers.LayerNormalization()(x)  # 添加規範層

x = (layers.LSTM(units=128, return_sequences=False))(x)
x = layers.Dropout(0.3)(x)

# 全連接層
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

# 定義輸出層，使用softmax激活函數
output_layer = layers.Dense(Y_train.shape[1], activation="linear")(x)
 # 創建並編譯模型
model_A = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model_A.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

 # 定義回調函數
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

 # 訓練模型
history_A=model_A.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32, callbacks=[reduce_lr, early_stopping])
 

predictions_A = model_A.predict(X_test)

predictions_A = scalerY.inverse_transform(predictions_A)
Y_test= scalerY.inverse_transform(Y_test)

A_MAE_LOSS_X=0
A_MAE_LOSS_Y=0

A_MSE_LOSS_X=0
A_MSE_LOSS_Y=0

for i in range(0,len(predictions_A)):
    A_MAE_LOSS_X=A_MAE_LOSS_X+abs(predictions_A[i][0]-Y_test[i][0])
    A_MAE_LOSS_Y=A_MAE_LOSS_Y+abs(predictions_A[i][1]-Y_test[i][1])
    
    A_MSE_LOSS_X=A_MSE_LOSS_X+(predictions_A[i][0]-Y_test[i][0])**2
    A_MSE_LOSS_Y=A_MSE_LOSS_Y+(predictions_A[i][1]-Y_test[i][1])**2
    
A_MAE_LOSS_X=A_MAE_LOSS_X/len(predictions_A)
A_MAE_LOSS_Y=A_MAE_LOSS_Y/len(predictions_A)

    
A_MSE_LOSS_X=A_MSE_LOSS_X/len(predictions_A)
A_MSE_LOSS_Y=A_MSE_LOSS_Y/len(predictions_A)
#model_A.summary()

# # Load the TensorBoard notebook extension

# court_min_x = 26.999998312177805
# court_max_x = 339.0000112887177
# court_min_y = 99.99998030598842
# court_max_y = 784.0000095285399
# double_center_x=(court_min_x + court_max_x)
# double_center_y=(court_min_y + court_max_y)

# for i in range(0,len(predictions_A)):
#     TEMPX=0
#     TEMPY=0
#     if (predictions_A[i][1]<442 and predictions_A[i][0]>=court_min_x and predictions_A[i][0]<105) or (predictions_A[i][1]>442 and predictions_A[i][0]<339 and predictions_A[i][0]>=261):
#         TEMPX=0
#         #count+=1
#     elif (predictions_A[i][1]<442 and predictions_A[i][0]>=105 and predictions_A[i][0]<261) or (predictions_A[i][1]>442 and predictions_A[i][0]<double_center_x-105 and predictions_A[i][0]>=double_center_x-261):
#         TEMPX=1
#         #count+=1
#     elif (predictions_A[i][1]<442 and predictions_A[i][0]>=261 and predictions_A[i][0]<339) or (predictions_A[i][1]>442 and predictions_A[i][0]<double_center_x-261 and predictions_A[i][0]>=double_center_x-339):
#         TEMPX=2
#         #count+=1
#     else:
#         TEMPX=3 # 出界
#     # elif predictions_A[i][1]<442 and  predictions_A[i][0]<=court_min_x:
#     #     TEMPX=3 # 出界
#     #     #count+=1
#     # else:
#     #     TEMPX=4 # 出界
#     #    # count+=1
    
    
#     if (predictions_A[i][1]>=court_min_y and predictions_A[i][1]<154) or (predictions_A[i][1]<double_center_y-court_min_y and predictions_A[i][1]>=double_center_y-154):
#          TEMPY=0
#         # count+=1
#     elif (predictions_A[i][1]>=154 and predictions_A[i][1]<322) or (predictions_A[i][1]<double_center_y-154 and predictions_A[i][1]>=double_center_y-322):
#          TEMPY=1
#          #count+=1
#     elif (predictions_A[i][1]>=322 and predictions_A[i][1]<442) or (predictions_A[i][1]<double_center_y-322 and predictions_A[i][1]>=double_center_y-442):
#          TEMPY=2
#         # count+=1
#     else:
#         TEMPY=3 # 出界
#         #count+=1   
#     predictions_A[i]=[TEMPX,TEMPY]
    
# for i in range(0,len(Y_test)):
#     TEMPX=0
#     TEMPY=0
#     if (Y_test[i][1]<442 and Y_test[i][0]>=court_min_x and Y_test[i][0]<105) or (Y_test[i][1]>442 and Y_test[i][0]<339 and Y_test[i][0]>=261):
#         TEMPX=0
#         #count+=1
#     elif (Y_test[i][1]<442 and Y_test[i][0]>=105 and Y_test[i][0]<261) or (Y_test[i][1]>442 and Y_test[i][0]<double_center_x-105 and Y_test[i][0]>=double_center_x-261):
#         TEMPX=1
#         #count+=1
#     elif (Y_test[i][1]<442 and Y_test[i][0]>=261 and Y_test[i][0]<339) or (Y_test[i][1]>442 and Y_test[i][0]<double_center_x-261 and Y_test[i][0]>=double_center_x-339):
#         TEMPX=2
#         #count+=1
#     else:
#         TEMPX=3 # 出界
#     # elif Y_test[i][1]<442 and  Y_test[i][0]<=court_min_x:
#     #     TEMPX=3 # 出界
#     #     #count+=1
#     # else:
#     #     TEMPX=4 # 出界
#     #    # count+=1
    
    
#     if (Y_test[i][1]>=court_min_y and Y_test[i][1]<154) or (Y_test[i][1]<double_center_y-court_min_y and Y_test[i][1]>=double_center_y-154):
#          TEMPY=0
#         # count+=1
#     elif (Y_test[i][1]>=154 and Y_test[i][1]<322) or (Y_test[i][1]<double_center_y-154 and Y_test[i][1]>=double_center_y-322):
#          TEMPY=1
#          #count+=1
#     elif (Y_test[i][1]>=322 and Y_test[i][1]<442) or (Y_test[i][1]<double_center_y-322 and Y_test[i][1]>=double_center_y-442):
#          TEMPY=2
#         # count+=1
#     else:
#         TEMPY=3 # 出界
#         #count+=1   
#     Y_test[i]=[TEMPX,TEMPY]
    
# P_X=predictions_A[:,0]
# P_Y=predictions_A[:,1]

# TEST_X=Y_test[:,0]
# TEST_Y=Y_test[:,1]
# count_X=0
# for i in range(0,len(P_X)):
#     if P_X[i]==TEST_X[i]:
#         count_X+=1
# count_Y=0
# for i in range(0,len(P_Y)):
#     if P_Y[i]==TEST_Y[i]:
#         count_Y+=1
        
# print("X預測準確率:",count_X/len(P_X))
# print("Y預測準確率:",count_Y/len(P_Y))

# cm = confusion_matrix(P_X, TEST_X, labels=range(4))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# #plt.xticks(rotation=45)  #X軸旋轉45度
# plt.title("predicted_A_X")
# plt.show()

# cm = confusion_matrix(P_Y, TEST_Y, labels=range(4))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# #plt.xticks(rotation=45)  #X軸旋轉45度
# plt.title("predicted_A_Y")
# plt.show()



Y_89=Y[:,8:10]
scalerY = MinMaxScaler(feature_range=(0, 1))
Y_89 = scalerY.fit_transform(Y_89)
#X_train=X[0:124,:,0:14]
#X_val=X[124:164,:,0:16]
#X_test=X[164:198,:,0:16]
X_train = X[0:740] #前四場當訓練集

X_val = X[740:890] #第五場當驗證集
X_test = X[890:] #第六場當測試集

# 將目標標籤同樣分成訓練集、驗證集和測試集
Y_train = Y_89[0:740]
Y_val = Y_89[740:890]
Y_test = Y_89[890:]

import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# 改进的模型架构
# 改进的模型架构
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.Dense(128, activation='relu')(input_layer)
attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
x = layers.Add()([x, attention_output])  # 殘差連接
x = layers.LayerNormalization()(x)  # 添加規範層

x = (layers.LSTM(units=128, return_sequences=False))(x)
x = layers.Dropout(0.3)(x)

# 全連接層
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

# 定義輸出層，使用softmax激活函數
output_layer = layers.Dense(Y_train.shape[1], activation="linear")(x)
 # 創建並編譯模型
model_B = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model_B.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

 # 定義回調函數
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

 # 訓練模型
history_B=model_B.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32, callbacks=[reduce_lr, early_stopping])

predictions_B = model_B.predict(X_test)

predictions_B = scalerY.inverse_transform(predictions_B)
Y_test= scalerY.inverse_transform(Y_test)

B_MAE_LOSS_X=0
B_MAE_LOSS_Y=0

B_MSE_LOSS_X=0
B_MSE_LOSS_Y=0

for i in range(0,len(predictions_B)):
    B_MAE_LOSS_X=B_MAE_LOSS_X+abs(predictions_B[i][0]-Y_test[i][0])
    B_MAE_LOSS_Y=B_MAE_LOSS_Y+abs(predictions_B[i][1]-Y_test[i][1])
    
    B_MSE_LOSS_X=B_MSE_LOSS_X+(predictions_B[i][0]-Y_test[i][0])**2
    B_MSE_LOSS_Y=B_MSE_LOSS_Y+(predictions_B[i][1]-Y_test[i][1])**2
    
B_MAE_LOSS_X=B_MAE_LOSS_X/len(predictions_B)
B_MAE_LOSS_Y=B_MAE_LOSS_Y/len(predictions_B)

    
B_MSE_LOSS_X=B_MSE_LOSS_X/len(predictions_B)
B_MSE_LOSS_Y=B_MSE_LOSS_Y/len(predictions_B)


plt.plot(history_B.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_B.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_B')



Y_1011=Y[:,10:12]
scalerY = MinMaxScaler(feature_range=(0, 1))
Y_1011 = scalerY.fit_transform(Y_1011)
#X_train=X[0:124,:,0:14]
#X_val=X[124:164,:,0:16]
#X_test=X[164:198,:,0:16]
X_train = X[0:740] #前四場當訓練集

X_val = X[740:890] #第五場當驗證集
X_test = X[890:] #第六場當測試集

# 將目標標籤同樣分成訓練集、驗證集和測試集
Y_train = Y_1011[0:740]
Y_val = Y_1011[740:890]
Y_test = Y_1011[890:]

import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# 改进的模型架构
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.Dense(128, activation='relu')(input_layer)
attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
x = layers.Add()([x, attention_output])  # 殘差連接
x = layers.LayerNormalization()(x)  # 添加規範層

x = (layers.LSTM(units=128, return_sequences=False))(x)
x = layers.Dropout(0.3)(x)

# 全連接層
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

# 定義輸出層，使用softmax激活函數
output_layer = layers.Dense(Y_train.shape[1], activation="linear")(x)
 # 創建並編譯模型
model_C = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model_C.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

 # 定義回調函數
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

 # 訓練模型
history_C=model_C.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32, callbacks=[reduce_lr, early_stopping])

predictions_C= model_C.predict(X_test)

predictions_C = scalerY.inverse_transform(predictions_C)
Y_test= scalerY.inverse_transform(Y_test)



C_MAE_LOSS_X=0
C_MAE_LOSS_Y=0

C_MSE_LOSS_X=0
C_MSE_LOSS_Y=0

for i in range(0,len(predictions_C)):
    C_MAE_LOSS_X=C_MAE_LOSS_X+abs(predictions_C[i][0]-Y_test[i][0])
    C_MAE_LOSS_Y=C_MAE_LOSS_Y+abs(predictions_C[i][1]-Y_test[i][1])
    
    C_MSE_LOSS_X=C_MSE_LOSS_X+(predictions_C[i][0]-Y_test[i][0])**2
    C_MSE_LOSS_Y=C_MSE_LOSS_Y+(predictions_C[i][1]-Y_test[i][1])**2
    
C_MAE_LOSS_X=C_MAE_LOSS_X/len(predictions_C)
C_MAE_LOSS_Y=C_MAE_LOSS_Y/len(predictions_C)

    
C_MSE_LOSS_X=C_MSE_LOSS_X/len(predictions_C)
C_MSE_LOSS_Y=C_MSE_LOSS_Y/len(predictions_C)

plt.plot(history_C.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_C.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_C')



Y_1213=Y[:,12:14]
scalerY = MinMaxScaler(feature_range=(0, 1))
Y_1213 = scalerY.fit_transform(Y_1213)
#X_train=X[0:124,:,0:14]
#X_val=X[124:164,:,0:16]
#X_test=X[164:198,:,0:16]
X_train = X[0:740] #前四場當訓練集

X_val = X[740:890] #第五場當驗證集
X_test = X[890:] #第六場當測試集

# 將目標標籤同樣分成訓練集、驗證集和測試集
Y_train = Y_1213[0:740]
Y_val = Y_1213[740:890]
Y_test = Y_1213[890:]

import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# 改进的模型架构
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.Dense(128, activation='relu')(input_layer)
attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
x = layers.Add()([x, attention_output])  # 殘差連接
x = layers.LayerNormalization()(x)  # 添加規範層

x = (layers.LSTM(units=128, return_sequences=False))(x)
x = layers.Dropout(0.3)(x)

# 全連接層
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

# 定義輸出層，使用softmax激活函數
output_layer = layers.Dense(Y_train.shape[1], activation="linear")(x)
 # 創建並編譯模型
model_D = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model_D.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

 # 定義回調函數
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

 # 訓練模型
history_D=model_D.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32, callbacks=[reduce_lr, early_stopping])

predictions_D= model_D.predict(X_test)

predictions_D = scalerY.inverse_transform(predictions_D)
Y_test= scalerY.inverse_transform(Y_test)


D_MAE_LOSS_X=0
D_MAE_LOSS_Y=0

D_MSE_LOSS_X=0
D_MSE_LOSS_Y=0

for i in range(0,len(predictions_D)):
    D_MAE_LOSS_X=D_MAE_LOSS_X+abs(predictions_D[i][0]-Y_test[i][0])
    D_MAE_LOSS_Y=D_MAE_LOSS_Y+abs(predictions_D[i][1]-Y_test[i][1])
    
    D_MSE_LOSS_X=D_MSE_LOSS_X+(predictions_D[i][0]-Y_test[i][0])**2
    D_MSE_LOSS_Y=D_MSE_LOSS_Y+(predictions_D[i][1]-Y_test[i][1])**2
    
D_MAE_LOSS_X=D_MAE_LOSS_X/len(predictions_D)
D_MAE_LOSS_Y=D_MAE_LOSS_Y/len(predictions_D)

    
D_MSE_LOSS_X=D_MSE_LOSS_X/len(predictions_D)
D_MSE_LOSS_Y=D_MSE_LOSS_Y/len(predictions_D)


plt.plot(history_D.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_D.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_D')








print("ball_type_acc:", acc / len(original_Y_test_labels))
print("next_ball_MAE:",next_ball_MAE_LOSS_X," ",next_ball_MAE_LOSS_Y)
print("next_ball_MSE:",next_ball_MSE_LOSS_X," ",next_ball_MSE_LOSS_Y)

print("A_MAE:",A_MAE_LOSS_X," ",B_MAE_LOSS_Y)
print("A_MSE:",A_MSE_LOSS_X," ",B_MSE_LOSS_Y)

print("B_MAE:",B_MAE_LOSS_X," ",B_MAE_LOSS_Y)
print("B_MSE:",B_MSE_LOSS_X," ",B_MSE_LOSS_Y)

print("C_MAE:",C_MAE_LOSS_X," ",C_MAE_LOSS_Y)
print("C_MSE:",C_MSE_LOSS_X," ",C_MSE_LOSS_Y)

print("D_MAE:",D_MAE_LOSS_X," ",D_MAE_LOSS_Y)
print("D_MSE:",D_MSE_LOSS_X," ",D_MSE_LOSS_Y)


# # Create TensorBoard log directory
plt.plot(history_shot_type.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_shot_type.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_shot_type')

plt.plot(history_next_ball.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_next_ball.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_next_ball_position')


plt.plot(history_A.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_A.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_A')

plt.plot(history_B.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_B.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_B')

plt.plot(history_C.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_C.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_C')

plt.plot(history_D.history['loss'], label=f'Train Fold {i + 1}')
plt.plot(history_D.history['val_loss'], label=f'Val Fold {i + 1}')
plt.title('history_D')

#12場地

M=[[1.5373038371601317, 0.6655617139970934, -802.1168258877335], [-0.013129641182294849, 6.431804504303463, -1551.498869990092], [-1.1510544606305043e-05, 0.003675563093109781, 1.0]]
p1=[438.6,273.2,1]
p2=[843.6,274.0,1]
p3=[988.6,659.0,1]
p4=[296.8,658.2,1]
temp=np.dot(M,p1)
min_x=temp[0]/temp[2]
min_y=temp[1]/temp[2]

temp=np.dot(M,p3)
max_x=temp[0]/temp[2]
max_y=temp[1]/temp[2]
print("min_x:",min_x)
print("max_x:",max_x)
print("min_y:",min_y)
print("max_y:",max_y)

#34場地
M=[[1.5573733896095519, 0.6789253402382057, -815.4992974842665], [-0.009690040188208928, 6.527490000670644, -1585.8966352778302], [-5.728654117601743e-06, 0.003734512710127637, 1.0]]
p1=[439.0,274.6,1]
p2=[844.0,275.2,1]
p3=[988.8,659.8,1]
p4=[296.4,658.8,1]
temp=np.dot(M,p1)
min_x=temp[0]/temp[2]
min_y=temp[1]/temp[2]

temp=np.dot(M,p3)
max_x=temp[0]/temp[2]
max_y=temp[1]/temp[2]
print("min_x:",min_x)
print("max_x:",max_x)
print("min_y:",min_y)
print("max_y:",max_y)

#56場地
M=[[1.5458761268559398, 0.6720814401496495, -807.4215187598494], [-0.016242136058745013, 6.476869150834237, -1561.5701964797506], [-1.1504151676562503e-05, 0.003707587791660278, 1.0]]	
p1=[438.6,273.2,1]
p2=[843.2,274.2,1]
p3=[988.4,659.0,1]
p4=[296.4,657.6,1]
temp=np.dot(M,p1)
min_x=temp[0]/temp[2]
min_y=temp[1]/temp[2]

temp=np.dot(M,p3)
max_x=temp[0]/temp[2]
max_y=temp[1]/temp[2]
print("min_x:",min_x)
print("max_x:",max_x)
print("min_y:",min_y)
print("max_y:",max_y)
#min_x: 26.999998312177805
#max_x: 339.0000112887177
#min_y: 99.99998030598842
#max_y: 784.0000095285399

for i in range(0,set6.shape[0]):
  if(set6.at[i, 'ball_type']=="擋小球" or set6.at[i, 'ball_type']=="放小球" or set6.at[i, 'ball_type']=="小平球"):
    set6.at[i, 'ball_type'] = '網前小球'
  elif(set6.iloc[i]['ball_type']=="挑球" or set6.iloc[i]['ball_type']=="防守回挑"):
    set6.at[i, 'ball_type'] = '挑球'
  elif(set6.iloc[i]['ball_type']=="防守回抽" or set6.iloc[i]['ball_type']=="平球" or set6.iloc[i]['ball_type']=="後場抽平球"):
    set6.at[i, 'ball_type'] = "平球"
    
for i in range(0,len(set6)):
    if(set6['ball_type'][i]=='網前小球'):
     set6['ball_type'][i]=1
    elif(set6['ball_type'][i]=='挑球'):
      set6['ball_type'][i]=2
    elif(set6['ball_type'][i]=='平球'):
     set6['ball_type'][i]=3
    elif(set6['ball_type'][i]=='推撲球'):
      set6['ball_type'][i]=4
    elif(set6['ball_type'][i]=='殺球'):
      set6['ball_type'][i]=5
    elif(set6['ball_type'][i]=='長球'):
      set6['ball_type'][i]=6
    elif(set6['ball_type'][i]=='切球'):
      set6['ball_type'][i]=7
    elif(set6['ball_type'][i]=='發長球'):
     set6['ball_type'][i]=8
    elif(set6['ball_type'][i]=='發短球'):
      set6['ball_type'][i]=9


import pygame,sys
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE = (  0,   0, 255)
GRAY = (200, 200, 200)
pygame.init()
screen_width, screen_height = 800, 600 #螢幕大小
screen  = pygame.display.set_mode((screen_width,screen_height)) #將screen 準備好
clock = pygame.time.Clock() #將clock準備好
FPS=60 #每秒60次更新螢幕
pygame.display.set_caption("badminton") #標題叫badminton
screen.fill((255,255,255)) #白色撲滿整個螢幕
court=pygame.image.load('court.png') #把court.png load進court這變數
court = pygame.transform.scale(court, (court.get_width() // 2, court.get_height()//2)) #court的長寬/2
court=pygame.transform.rotate(court,90) #court旋轉90度，讓他變成直的
#badminton_coords = [(78, 121), (324, 121), (78, 772), (324, 772)] #原先紀錄羽毛球場上的四個角的座標(前面min MAX的)


badminton_coords = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)] #原先紀錄羽毛球場上的四個角的座標


pygame_coords = [(300, 40), (535, 40), (300, 560), (535, 560)] # pygame畫布上對應的四個角的座標

# 計算縮放比例
scale_x = (pygame_coords[1][0] - pygame_coords[0][0]) / (badminton_coords[1][0] - badminton_coords[0][0])
scale_y = (pygame_coords[3][1] - pygame_coords[0][1]) / (badminton_coords[3][1] - badminton_coords[0][1])

# 計算偏移量
offset_x = pygame_coords[0][0] - badminton_coords[0][0] * scale_x
offset_y = pygame_coords[0][1] - badminton_coords[0][1] * scale_y

right_button_x=180
right_button_y=50
right_button_width, right_button_height = 50, 50
button_font = pygame.font.SysFont(None, 36)
right_button_text = button_font.render("->", True, BLACK)
right_button = pygame.Rect(right_button_x, right_button_y, right_button_width, right_button_height)

left_button_x=100
left_button_y=50
left_button_width,left_button_height = 50, 50
button_font = pygame.font.SysFont(None, 36)
left_button_text = button_font.render("<-", True, BLACK)
left_button = pygame.Rect(left_button_x, left_button_y, left_button_width, left_button_height)






# court_x_min, court_x_max = 121, 772 
# court_y_min, court_y_max = 78, 324
# pygame_x_min,pygame_x_max=29,770 
# pygame_y_min,pygame_y_max=132,468
#print(court.get_width() // 1.4)
#print(court.get_height()//1.4 )


def draw_player(set,player,count): #set代表你要撥放哪一個set
    if(player=='A'):
        player_coords = (set["player_A_x"][count], set["player_A_y"][count])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        if(set["player"][count]=='A'): # 如果打球的人是A
            pygame.draw.circle(screen, GREEN, pygame_player_coords, 10, 0) #r=10 綠色
        else:
            pygame.draw.circle(screen, WHITE, pygame_player_coords, 10, 0) #r=10 白色
        font = pygame.font.Font("freesansbold.ttf", 20)  # 使用字型，大小20
        text_surface = font.render("A", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
    elif(player=='B'):
        player_coords = (set["player_B_x"][count], set["player_B_y"][count])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        if(set["player"][count]=='B'):# 如果打球的人是B
            pygame.draw.circle(screen, GREEN, pygame_player_coords, 10, 0) #r=10 綠色
        else:
            pygame.draw.circle(screen, WHITE, pygame_player_coords, 10, 0) #r=10 白色
        font = pygame.font.Font("freesansbold.ttf", 20)  # 使用字型，大小20
        text_surface = font.render("B", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
    elif(player=='C'):
        player_coords = (set["player_C_x"][count], set["player_C_y"][count])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        if(set["player"][count]=='C'): # 如果打球的人是C
            pygame.draw.circle(screen, GREEN, pygame_player_coords, 10, 0) #r=10 綠色
        else:
            pygame.draw.circle(screen, WHITE, pygame_player_coords, 10, 0) #r=10 白色
        font = pygame.font.Font("freesansbold.ttf", 20)  # 使用字型，大小20
        text_surface = font.render("C", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
    elif(player=='D'):
        player_coords = (set["player_D_x"][count], set["player_D_y"][count])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        if(set["player"][count]=='D'): # 如果打球的人是D
            pygame.draw.circle(screen, GREEN, pygame_player_coords, 10, 0) #r=10 綠色
        else:
            pygame.draw.circle(screen, WHITE, pygame_player_coords, 10, 0) #r=10 白色
        font = pygame.font.Font("freesansbold.ttf", 20)  # 使用字型，大小20
        text_surface = font.render("D", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
    elif(player=='@'):
        player_coords = (set["hit_x"][count], set["hit_y"][count])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        pygame.draw.circle(screen, WHITE, pygame_player_coords, 10, 0)
        font = pygame.font.Font("freesansbold.ttf", 15)  # 使用字型，大小20
        text_surface = font.render("@", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
    elif(player=='*'):
        player_coords = (set["return_x"][count], set["return_y"][count])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        pygame.draw.circle(screen, RED, pygame_player_coords, 10, 0)
        font = pygame.font.Font("freesansbold.ttf", 15)  # 使用字型，大小20
        text_surface = font.render("T", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)


def draw_prediction_A(count):
        player_coords = (predictions_A[count][0], predictions_A[count][1])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        pygame.draw.circle(screen,  (164,219,236), pygame_player_coords, 10, 0) #r=10 綠色
        font = pygame.font.Font("freesansbold.ttf", 17)  # 使用字型，大小20
        text_surface = font.render("AP", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)

def draw_prediction_B(count):
        player_coords = (predictions_B[count][0], predictions_B[count][1])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        pygame.draw.circle(screen,  (164,219,236), pygame_player_coords, 10, 0) #r=10 綠色
        font = pygame.font.Font("freesansbold.ttf", 17)  # 使用字型，大小20
        text_surface = font.render("BP", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
def draw_prediction_C(count):
        player_coords = (predictions_C[count][0], predictions_C[count][1])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        pygame.draw.circle(screen,  (164,219,236), pygame_player_coords, 10, 0) #r=10 綠色
        font = pygame.font.Font("freesansbold.ttf", 17)  # 使用字型，大小20
        text_surface = font.render("CP", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
def draw_prediction_D(count):
        player_coords = (predictions_D[count][0], predictions_D[count][1])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        pygame.draw.circle(screen, (164,219,236), pygame_player_coords, 10, 0) #r=10 綠色
        font = pygame.font.Font("freesansbold.ttf", 17)  # 使用字型，大小20
        text_surface = font.render("DP", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
        
def draw_prediction_next_ball(count):
        player_coords = (predictions_next_ball[count][0], predictions_next_ball[count][1])
        pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        pygame.draw.circle(screen,  (164,219,236), pygame_player_coords, 10, 0) #r=10 綠色
        font = pygame.font.Font("freesansbold.ttf", 17)  # 使用字型，大小20
        text_surface = font.render("NB", True, BLACK)  # 渲染文字到表面上
        text_rect = text_surface.get_rect(center=pygame_player_coords)  # 將文字居中對齊到圓的中心
        screen.blit(text_surface, text_rect)
def draw_prediction_shot_type(count):
        fontObj = pygame.font.Font('freesansbold.ttf', 32)    
        #player_coords = (predictions_next_ball[count][0], predictions_next_ball[count][1])
       # pygame_player_coords = (int(player_coords[0] * scale_x + offset_x), int(player_coords[1] * scale_y + offset_y))
        #pygame.draw.circle(screen, (128,0,128), pygame_player_coords, 10, 0) #r=10 綠色
        temp=model_shot_type_predict[count]
        if(temp==1):
            temp='Net shot'# 網前小球
        elif(temp==2):
            temp='lift' #挑球
        elif(temp==3):
            temp='Drive' #平球
        elif(temp==4):
            temp='Push Shot' #推撲球
        elif(temp==5):
            temp='Smash' #殺球
        elif(temp==6):
            temp='Clear' #長球
        elif(temp==7):
            temp='Smash' #切球
        elif(temp==8):
            temp='Long Serve' #發長球
        elif(temp==9):
            temp='Short Serve' #發短球
        

        textSurfaceObj = fontObj.render(temp, True, (0, 0, 255),WHITE) #創文字Surface
        textRectObj = textSurfaceObj.get_rect() #文字方塊
        textRectObj.center = (650, 300)
        screen.blit(textSurfaceObj, textRectObj)
        
        #predict_shot_type文字
        textSurfaceObj = fontObj.render("Predict:", True, (0, 0, 255),WHITE)     
        textRectObj = textSurfaceObj.get_rect() #文字方塊
        textRectObj.center = (650, 250)
        screen.blit(textSurfaceObj, textRectObj)

    
def draw_board(count):
#ball_round
    fontObj = pygame.font.Font('freesansbold.ttf', 32)    
    temp=set6["ball_round"][count]
    textSurfaceObj = fontObj.render(str(temp), True, (0, 0, 255),WHITE) #創文字Surface
    textRectObj = textSurfaceObj.get_rect() #文字方塊
    textRectObj.center = (230, 200)
    screen.blit(textSurfaceObj, textRectObj)

#ball_round文字
    textSurfaceObj = fontObj.render("Ball_Round:", True, (0, 0, 255),WHITE)     
    textRectObj = textSurfaceObj.get_rect() #文字方塊
    textRectObj.center = (100, 200)
    screen.blit(textSurfaceObj, textRectObj)

#rally
    fontObj = pygame.font.Font('freesansbold.ttf', 32)    
    temp=set6["rally"][count]
    textSurfaceObj = fontObj.render(str(temp), True, (0, 0, 255),WHITE) #創文字Surface
    textRectObj = textSurfaceObj.get_rect() #文字方塊
    textRectObj.center = (230, 150)
    screen.blit(textSurfaceObj, textRectObj)

#rally文字
    textSurfaceObj = fontObj.render("Rally:", True, (0, 0, 255),WHITE)     
    textRectObj = textSurfaceObj.get_rect() #文字方塊
    textRectObj.center = (100, 150)
    screen.blit(textSurfaceObj, textRectObj)
#shot_type
    fontObj = pygame.font.Font('freesansbold.ttf', 32)    
    temp=set6["ball_type"][count]
    if(temp==1):
        temp='Net shot'
    elif(temp==2):
        temp='lift'
    elif(temp==3):
        temp='Drive'
    elif(temp==4):
        temp='Push Shot'
    elif(temp==5):
        temp='Smash'
    elif(temp==6):
        temp='Clear'
    elif(temp==7):
        temp='Smash'
    elif(temp==8):
        temp='Long Serve'
    elif(temp==9):
        temp='Short Serve'
    textSurfaceObj = fontObj.render(temp, True, (0, 0, 255),WHITE) #創文字Surface
    textRectObj = textSurfaceObj.get_rect() #文字方塊
    textRectObj.center = (150, 300)
    screen.blit(textSurfaceObj, textRectObj)

#shot_type文字
    textSurfaceObj = fontObj.render("Shot_Type:", True, (0, 0, 255),WHITE)     
    textRectObj = textSurfaceObj.get_rect() #文字方塊
    textRectObj.center = (100, 250)
    screen.blit(textSurfaceObj, textRectObj)





screen.blit(court,(300,40))
count=0

INDEX = []

# 遍歷每個 rally
for rally, group in set6.groupby('rally'):
    # 篩選出 ball_round 大於 5 的行
    filtered_group = group[group['ball_round'] > 5]
    # 將這些行的索引加入到 INDEX 中
    INDEX.extend(filtered_group.index)

prediction_count=0
#INDEX_count=0
while True:
    for event in pygame.event.get():
        if count==INDEX[prediction_count]:
            draw_prediction_A(prediction_count)
            draw_prediction_B(prediction_count)
            draw_prediction_C(prediction_count)
            draw_prediction_D(prediction_count)
            draw_prediction_next_ball(prediction_count)
            draw_prediction_shot_type(prediction_count)
        draw_board(count)
        draw_player(set6,'A',count)
        draw_player(set6,'B',count)
        draw_player(set6,'C',count)
        draw_player(set6,'D',count)
        draw_player(set6,"@",count) #hit_ball
        draw_player(set6,"*",count) #return_ball
       # draw_score(count)
        #畫場上狀態、得分
        if event.type == pygame.MOUSEBUTTONDOWN:
            #print(pygame.mouse.get_pos())
            if right_button.collidepoint(event.pos): # 檢查是否點擊了按鈕
                screen.fill((255, 255, 255))
                screen.blit(court, (300, 40))
                if count==INDEX[prediction_count]:
                    prediction_count+=1
                count+=1
                #print("count:",count)
                #print("prediction_count:",prediction_count)
               # prediction_count+=1
            elif left_button.collidepoint(event.pos) and count>0: # 檢查是否點擊了按鈕
                screen.fill((255, 255, 255))
                screen.blit(court, (300, 40))
                if count-1==INDEX[prediction_count-1]:
                    prediction_count-=1
                count-=1
              #  print("count:",count)
              #  print("prediction_count:",prediction_count)
                #prediction_count-=1
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
            
   
    pygame.draw.rect(screen, GRAY, right_button)
    screen.blit(right_button_text, (right_button_x+20,right_button_y+10))
    pygame.draw.rect(screen, GRAY, left_button)
    screen.blit(left_button_text, (left_button_x+20,left_button_y+10))     
    clock.tick(FPS)
    pygame.display.update()