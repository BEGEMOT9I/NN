# -*- coding: utf-8 -*-

from mnist import MNIST
import numpy as np

### Функция активации (rectified linear unit) ###
def ReLU(x):
  return np.maximum(x, 0)

### Функция предсказания числа на картинке (Сумматор) ###
def PredictImage(img, w):
  resp = list(range(0, 10))
  for i in range(0,10):
    r = w[i] * img
    r = ReLU(np.sum(r) + b[i])
    resp[i] = r

  return np.argmax(resp)

'''
Загружаем пикчи. Сами картинки размером 28х28, но в наборе
они представлены одномерным массивом длиной в 784 символа
'''
mndata = MNIST("./data")
tr_images, tr_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

### Нормализация тестовых и тренировочных картинок ###
for i in range(0, len(test_images)):
  test_images[i] = np.array(test_images[i], dtype="float") / 255
     
for i in range(0, len(tr_images)):
  tr_images[i] = np.array(tr_images[i], dtype="float") / 255
######################################################


### Генерируем рандомные веса и коэффициент смещения (биас) ###
w = (2 * np.random.rand(10, 784) - 1) / 10
b = (2 * np.random.rand(10) - 1) / 10
###############################################################
 
for n in range(len(tr_images)):
  img = tr_images[n]
  cls = tr_labels[n]

  ### Вектор вероятностей того, что на картинке изображено какое-то число ###
  resp = np.zeros(10, dtype=np.float32)
  for i in range(0,10):
    r = w[i] * img
    r = ReLU(np.sum(r) + b[i])
    resp[i] = r
  ###########################################################################

  '''
  Находим максимальное значение вероятности; обнуляем остальные
  значения массива и наибольшую вероятность заменяем 1
  '''
  resp_cls = np.argmax(resp)
  resp = np.zeros(10, dtype=np.float32)
  resp[resp_cls] = 1.0

  ### Какое же на самом деле было число ###
  true_resp = np.zeros(10, dtype=np.float32)
  true_resp[cls] = 1.0
  #########################################

  '''
  Находим ошибку и высчитываем новые значения весовых коэффициентов
  и новое значения для коэффициента смещения
  '''
  error = resp - true_resp
  delta = error * ((resp >= 0) * np.ones(10))
  for i in range(0,10):
    w[i] -= np.dot(img, delta[i])
    b[i] -= delta[i]

total = len(test_images)
valid = 0
invalid = []

for i in range(0, total):
  img = test_images[i]
  predicted = PredictImage(img, w)
  true = test_labels[i]
  if predicted == true:
    valid = valid + 1
  else:
    invalid.append({"image":img, "predicted":predicted, "true":true})

print("Точность: %s%%" % (float(valid) / total * 100))