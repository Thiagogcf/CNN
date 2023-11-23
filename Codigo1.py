import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense
from PIL import Image
from keras.applications import VGG16, VGG19, ResNet50
from keras.preprocessing.image import ImageDataGenerator

# Definindo os diretórios de dados
train_dir = 'data/train'
test_dir = 'data/test'

# Função para listar imagens de gatos e cachorros
def list_images(directory):
    cats_dir = os.path.join(directory, 'cats')
    dogs_dir = os.path.join(directory, 'dogs')

    cat_images = [os.path.join(cats_dir, filename) for filename in os.listdir(cats_dir)]
    dog_images = [os.path.join(dogs_dir, filename) for filename in os.listdir(dogs_dir)]

    return cat_images, dog_images

# Função de pré-processamento das imagens
def preprocess_images(image_paths, target_size=(150, 150)):
    images = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')  # Convertendo para RGB
        img = img.resize(target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.0  # Normalização dos valores dos pixels
        images.append(img)
    return images


# Listando e pré-processando as imagens de treino e teste
train_cat_images, train_dog_images = list_images(train_dir)
test_cat_images, test_dog_images = list_images(test_dir)

train_cat_images = preprocess_images(train_cat_images)
train_dog_images = preprocess_images(train_dog_images)
test_cat_images = preprocess_images(test_cat_images)
test_dog_images = preprocess_images(test_dog_images)

# Criando labels e combinando os dados
train_X = np.array(train_cat_images + train_dog_images)
train_y = np.array([0] * len(train_cat_images) + [1] * len(train_dog_images))

test_X = np.array(test_cat_images + test_dog_images)
test_y = np.array([0] * len(test_cat_images) + [1] * len(test_dog_images))

# Modelo com VGG16 como base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilação do modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Aplicar aumento de dados
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow(train_X, train_y, batch_size=32)

# Treinando o modelo
model.fit(train_generator, epochs=20, validation_data=(test_X, test_y))

# Avaliando o modelo
test_loss, test_acc = model.evaluate(test_X, test_y)
print(f'Acurácia no conjunto de teste: {test_acc * 100:.2f}%')
