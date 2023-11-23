import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

# Certificando que o TensorFlow vai usar a GPU
with tf.device('/GPU:0'):
    # Definir o caminho para os dados
    train_data_path = './data/train'  # Substitua pelo caminho correto para os dados de treino
    test_data_path = './data/test'    # Substitua pelo caminho correto para os dados de teste

    # Criar um gerador de dados de imagem
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Preparar o gerador de dados de treinamento e teste
    training_set = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

    test_set = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

    # Ajustar o número de passos por época para o tamanho do conjunto de dados
    steps_per_epoch = len(training_set)
    validation_steps = len(test_set)

    # Construir uma CNN padrão
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treinar a CNN padrão
    model.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=25,
        validation_data=test_set,
        validation_steps=validation_steps)

    # Substituir por uma rede especializada - VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Treinar a rede especializada
    model.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=25,
        validation_data=test_set,
        validation_steps=validation_steps)
