import os
import json
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)
BATCH_SIZE = 10 

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,      
    width_shift_range=0.2,  
    height_shift_range=0.2, 
    shear_range=0.2,        
    zoom_range=0.2,        
    horizontal_flip=True,   
    fill_mode='nearest',
    validation_split=0.2    
)

train_generator = train_datagen.flow_from_directory(
    'dataset', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

labels = {v: k for k, v in train_generator.class_indices.items()}
with open("labels.json", "w") as f:
    json.dump(labels, f)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = True 

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) 
x = Dense(128, activation='relu')(x)
predictions = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=base.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.00001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Training with Augmentation... This will take longer but is smarter.")
model.fit(train_generator, epochs=10, validation_data=validation_generator)

model.save('food_model.h5')
print("Done!")