import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as xet

from glob import glob
from keras.models import Model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.applications import InceptionResNetV2
from keras.layers import Dense, Flatten, Input
from keras.preprocessing.image import load_img, img_to_array

# Создаем шаблон поиска файлов
path = glob('./images/*.xml')

# Создаем словарь который будет содержать в себя путь к файлу
# А так же значения переменных по которым будет находится табличка с номером автомобися
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])

# Заполняем наш словарь данными из xml файла
for filename in path:
    # Данные из xml файла заносим в переменную info и получаем дерево
    info = xet.parse(filename)
    root = info.getroot()

    # В этом дереве мы ищем тэг object в котором содержатся данные о местоположение таблички с номером автомобиля
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    # Вставляем полученные данные в наш словарь
    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

# В словаре мы имеем формат ключ-значение, на его основе создаем DataFrame
# Таблица в которой каждая строка будет содержать в себе путь к файлу и данные об аннотациях
df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv',index=False)

# Для примера отображаем как это выглядит
df.head()

# Создаем функцию чтобы на основе файлов xml мы получали данные о картинках которые соответвуют каждому xml файлу
# Если в краце то у каждой анотации есть тэг filename в котором содержиться название файла на который ссылается аннотация
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images',filename_image)
    return filepath_image

# Создаем массив который содержит в себе путь к всем изображениям
image_path = list(df['filepath'].apply(getFilename))

# Самый тяжелый процесс в данном случае, это нормализация данных, которая происходит в приведенном ниже коде
# Нормализация данных нужна чтобы ускорить работу нейронной сети и более стабильного обучения
# Я привел только несколько причин, зачем надо это делать
labels = df.iloc[:,1:].values
data = []
output = []

for ind in range(len(image_path)):
    # Мы должны нормализовать данные для всех изображений
    # В данном фрагменте кода мы будем проходить по всем изображениям
    # Считывать изображения с помощью OpenCV, получаем данные, высоты, ширины, и количество каналов
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    
    # Здесь мы считываем изображения благодаря функциям в keras, с фиксированным размером 224x224
    # Так как наша предобученная нейронная сеть работает с изображениями фиксированного размера 224x224
    # Делением на 255 значений всех пикселей мы нормалузием данные
    load_image = load_img(image, target_size = (224, 224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr / 255.0 
    
    # Считывем аннотации для каждого изображения и делим их на ширину с высотой соответсвенно
    # И записываем их в виде кортежа
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin / w, xmax / w
    nymin,nymax = ymin / h, ymax / h
    label_norm = (nxmin, nxmax, nymin, nymax)
    
    # В data записываем нормализованное изображение
    # В output нормализованные данные меток
    data.append(norm_load_image_arr)
    output.append(label_norm)


# Конвертируем данные в массив с типом float32
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

# Разделяем данные на обучающие и тестируемые
x_train,x_test,y_train,y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)

# Мы будем использовать предобученную модель, чтобы обучить ее на наших данных
# Веса модели загружены на наборе данных imagenet
# Так же мы указываем что, верхний слой модели загружать не надо, так как мы его заменим своим
# И послдений параметр говорит о том какие будут входные данные
inception_resnet = InceptionResNetV2(weights = "imagenet", include_top = False, input_tensor = Input(shape = (224, 224, 3)))

# Создаем новый верхний слой модели основываясь на наших данных
# inception_resnet.output сохраняет выходной тензор модели
# Перед подключением полносвязных слоев, добавляем слой Flatten, который преобразует выходной тензор в одномерный вектор
# Выходной слой содержит 4 нейрона, так как нам нужны 4 координаты
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500, activation = "relu")(headmodel)
headmodel = Dense(250, activation = "relu")(headmodel)
headmodel = Dense(4, activation = 'sigmoid')(headmodel)

# Создаем новую модель на основе общей
# Соединяем входной и выходной тензоры
model = Model(inputs = inception_resnet.input, outputs = headmodel)

# Компилируем нашу модель
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=["accuracy"])

# Создаем TensorBoard в котором будут хранится логи обучения и обучаем модель
# Не рекомендую обучать модель на компьютере так как займет слишком много времени
# При обучении использовался colab что бы получить доступ к мощностям google
tfb = TensorBoard('object_detection')
model.fit(x=x_train,y=y_train,batch_size=10,epochs=180,
          validation_data=(x_test,y_test), verbose = 1, callbacks=[tfb])

# Сохраняем модель после обучения
model.save('./model.h5')