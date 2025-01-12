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
      combined_data['ball_type'][i]=7 #交換位置
    elif(combined_data['ball_type'][i]=='切球'):
      combined_data['ball_type'][i]=6#交換位置
    elif(combined_data['ball_type'][i]=='發長球'):
     combined_data['ball_type'][i]=8
    elif(combined_data['ball_type'][i]=='發短球'):
      combined_data['ball_type'][i]=9
      
combined_data = combined_data.iloc[:,[1, 2, 8, 10,16,17,18,19,20,21,22,23,24,25]] #加入return_x return_y

X_data_list = []
count=0
for i in range(0,combined_data.shape[0]-1):
  if(combined_data['rally'][i]!=combined_data['rally'][i+1]):
    X_sample_list = []
    for j in range(count,i+1):
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
window_size = 1  # 設定滑動視窗大小

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
      row_list[5:5] =[0,0,0,0,0,0,1,0,0]
    elif(shot_type==8):
      row_list[5:5] = [0,0,0,0,0,0,0,1,0]
    elif(shot_type==9):
      row_list[5:5] = [0,0,0,0,0,0,0,0,1]
    temp.append(row_list)

X=temp


X=np.array(X)
Y=np.array(Y)

X=X.reshape(int(len(X)/window_size),window_size,24)

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
# scalerX = MinMaxScaler(feature_range=(0, 1))



# for i in range(X.shape[0]):  # 遍历每个数组
#   for j in range(X.shape[1]):  # 遍历数组中的每个元素
#     X[i, j][14:24] = scalerX.fit_transform(X[i, j][14:24].reshape(-1, 1)).reshape(-1)  # 特征缩放

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

# 將數據集分成訓練集、驗證集和測試集
#X=X[:,:,1:14]  #這行程式碼代表不考慮球員的位置和回合數

X_train = X[0:int(len(X)/10*8)]

X_val = X[int(len(X)/10*8):int(len(X)/10*9)] #第五場當驗證集

X_test = X[int(len(X)/10*9):] #第六場當測試集

# 將目標標籤同樣分成訓練集、驗證集和測試集
Y_train = Y_shot_type[0:int(len(X)/10*8)]
Y_val = Y_shot_type[int(len(X)/10*8):int(len(X)/10*9)]
Y_test = Y_shot_type[int(len(X)/10*9):]


# 匯入TensorFlow和Keras相關庫
import tensorflow as tf
from tensorflow.keras import layers, Input, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_history(history_list):
        plt.figure(figsize=(12, 6))
    #for i, history in enumerate(history_list):
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label=f'Train Fold {i + 1}')
        plt.plot(history.history['val_loss'], label=f'Val Fold {i + 1}')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])))
        
        plt.subplot(2, 1, 2)
        plt.plot(history.history['accuracy'], label=f'Train Fold {i + 1}')
        plt.plot(history.history['val_accuracy'], label=f'Val Fold {i + 1}')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.ylim(0, 1)
    
        plt.tight_layout()
        plt.show()

Y_train_labels = np.argmax(Y_train, axis=1)
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train_labels), y=Y_train_labels)
# class_weights_dict = dict(enumerate(class_weights))
# max_weight = 1.3
# min_weight = 0.1

# adjusted_class_weights = {k: min(max_weight, max(min_weight, v)) for k, v in class_weights_dict.items()}
adjusted_class_weights ={0: 6.1/10,
  1: 8.9/10,
  2: 3.2/10,
  3: 4.1/10,
  4: 6.7/10,
  5: 10/10,
  6: 10/10}
# train:({3: 41,
#         4: 32, 
#         1: 22, 
#         5: 20, 
#         2: 15, 
#         7: 3, 
#         6: 1})
# 30
# val:({3: 10,
#       4: 6, 
#       5: 5, 
#       1: 5,
#       7: 2, 
#       2: 2})
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
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
#將球員位置切分成六塊
# tempX=[]
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         temp=[]
#         for k in range(1):
#             odd_value= X[i][j][14+2*k]
#             even_value=X[i][j][14+2*k+1]
#             if (even_value < 442 and odd_value < 183 or(even_value > 442 and odd_value > 183) ):
#                 TEMPX = 0 #左邊
#             else:
#                 TEMPX = 1  # 右邊

#             # 判斷 TEMPY
#             if (even_value  < 270) or(even_value>614):
#                 TEMPY = 0
#             else:
#                 TEMPY = 1  # 遠離球場
                
#             if TEMPX == 0 and TEMPY == 0 and even_value<442:
#                 temp.append([1,0,0,0,0,0,0,0])
                
#             elif TEMPX == 0 and TEMPY == 1 and even_value<442:
#                 temp.append([0,1,0,0,0,0,0,0])
                
#             elif TEMPX == 1 and TEMPY == 0 and even_value<442:
#                 temp.append([0,0,1,0,0,0,0,0])
                
#             elif TEMPX == 1 and TEMPY == 1 and even_value<442:
#                 temp.append([0,0,0,1,0,0,0,0])
            
#             if TEMPX == 0 and TEMPY == 0 and even_value>442:
#                 temp.append([0,0,0,0,1,0,0,0])
                
#             elif TEMPX == 0 and TEMPY == 1 and even_value>442:
#                 temp.append([0,0,0,0,0,1,0,0])
                
#             elif TEMPX == 1 and TEMPY == 0 and even_value>442:
#                 temp.append([0,0,0,0,0,0,1,0])
                
#             elif TEMPX == 1 and TEMPY == 1 and even_value>442:
#                 temp.append([0,0,0,0,0,0,0,1])
                
#         flattened_temp = list(itertools.chain(*temp))
#         tempX.append(np.concatenate((X[i][j][0:14], flattened_temp)))

tempX=[]
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        temp=[]
        for k in range(1):
            odd_value= X[i][j][14+2*k]
            even_value=X[i][j][14+2*k+1]
            if (even_value < 442 and odd_value < 183 or(even_value > 442 and odd_value > 183) ):
                TEMPX = 0 #左邊
            else:
                TEMPX = 1  # 右邊

            # 判斷 TEMPY
            if (even_value  < 270) or(even_value>614):
                TEMPY = 0
            else:
                TEMPY = 1  # 遠離球場
                
            if TEMPX == 0 and TEMPY == 0 :
                temp.append([1,0,0,0])
                
            elif TEMPX == 0 and TEMPY == 1 :
                temp.append([0,1,0,0])
                
            elif TEMPX == 1 and TEMPY == 0 :
                temp.append([0,0,1,0])
                
            elif TEMPX == 1 and TEMPY == 1 :
                temp.append([0,0,0,1])
            
                
        flattened_temp = list(itertools.chain(*temp))
        tempX.append(np.concatenate((X[i][j][0:14], flattened_temp)))

X=tempX
X=np.array(X)
X=X.reshape(int(len(X)/window_size),window_size,X.shape[1])
from sklearn.utils import shuffle
#X=X[:,:,5:]
#
#X=X[:,:,1:14]
#X=X[:,:,1:]
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y_shot_type[train_index], Y_shot_type[val_index]
    #X_train, Y_train = shuffle(X_train, Y_train) #對每個fold 做shuffle
    #X_val, Y_val = shuffle(X_val, Y_val)
    #計算類別權重
    Y_train_labels = np.argmax(Y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train_labels), y=Y_train_labels)
    class_weights_dict = dict(enumerate(class_weights))
    max_weight = 2.6
    min_weight = 1

    adjusted_class_weights = {k: min(max_weight, max(min_weight, v)) for k, v in class_weights_dict.items()}

    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = layers.Dense(128, activation='relu')(input_layer)
    attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    #x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=False))(x)
    #x = (layers.LSTM(units=128, return_sequences=True))(x)
    #x = layers.Dropout(0.3)(x)
    x = (layers.LSTM(units=128, return_sequences=False))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    output_layer = layers.Dense(Y_train.shape[1], activation="softmax")(x)
    model_shot_type = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.001)
    model_shot_type.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #callbacks=[reduce_lr, early_stopping]
    history = model_shot_type.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=64, class_weight=adjusted_class_weights, callbacks=[reduce_lr, early_stopping])
    #history = model_shot_type.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32, callbacks=[reduce_lr, early_stopping])
    history_list.append(history)
    #history_list[0].history
    # 保存validation loss和accuracy
    val_loss_list.append(history.history['val_loss'][-1])
    train_loss_list.append(history.history['loss'][-1])

    predict_labels = model_shot_type.predict(X_val)
    predict_labels = np.argmax(predict_labels, axis=1)
    original_Y_val_labels = np.argmax(Y_val, axis=1)
    acc = np.sum(predict_labels == original_Y_val_labels) / len(original_Y_val_labels)
    print("acc:",acc)
    accs.append(acc)

    # 計算並打印 Confusion Matrix
    cm = confusion_matrix(original_Y_val_labels, predict_labels, labels=range(num_classes))
    #custom_labels = ['網前小球', '挑球', '平球', '推撲球', '殺球', '長球','切球']
    custom_labels = ['網前小球', '挑球', '平球', '推撲球', '殺球','切球']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=custom_labels)
    disp.plot(cmap=plt.cm.Blues)
    #plt.xticks(rotation=45)  #X軸旋轉45度
    plt.show()
    total_cm += cm
    plot_history(history)

#custom_labels = ['網前小球', '挑球', '平球', '推撲球', '殺球', '長球','切球']
custom_labels = ['網前小球', '挑球', '平球', '推撲球', '殺球','切球']
disp = ConfusionMatrixDisplay(confusion_matrix=total_cm,display_labels=custom_labels)
#fig, ax = plt.subplots(figsize=(8, 8))  # 可以調整圖形大小
disp.plot(cmap=plt.cm.Blues, values_format='0.0f')
#plt.xticks(rotation=45)  #X軸旋轉45度
plt.title('Total Confusion Matrix')
plt.show()

print("平均准确率:", np.mean(accs))
# 计算平均validation loss和average_train_loss

average_train_loss=np.mean(train_loss_list)

average_val_loss = np.mean(val_loss_list)

# 打印平均validation loss和accuracy

print("平均 train Loss:", average_train_loss)

print("平均 Validation Loss:", average_val_loss)
    
def plot_all_history(history_list):
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(history_list):
        plt.subplot(2, 1, 1)
        #plt.plot(history.history['loss'], label=f'Train Fold {i + 1}')
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
       # plt.plot(history.history['val_loss'], label=f'Val Fold {i + 1}')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 1, 2)
       # plt.plot(history.history['accuracy'], label=f'Train Fold {i + 1}')
       # plt.plot(history.history['val_accuracy'], label=f'Val Fold {i + 1}')
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
        plt.tight_layout()
        plt.show()