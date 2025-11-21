import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carregar modelo pré-treinado sem a parte final (top)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False  # congela as camadas

# Construir novo modelo
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # duas classes: gato e cachorro
])

# Compilar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparar dados (exemplo com ImageDataGenerator)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'dataset/cats_vs_dogs',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/cats_vs_dogs',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Treinar
history = model.fit(train_data, validation_data=val_data, epochs=5)