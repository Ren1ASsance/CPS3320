import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

# 检查GPU可用性 <--
print("Checking GPU availability...")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("GPU will be used for training")
    # 设置GPU内存增长 <--
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU available, using CPU")

# 设置路径 <--
data_dir = 'animal_dataset/dataset'  # 包含80个子文件夹的根目录
log_dir = 'training_logs'
checkpoint_dir = 'training_checkpoints'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 参数设置
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 50
NUM_CLASSES = 80  # 80种动物

# 创建数据集 <--
print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

# 数据预处理和增强
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# 构建CNN模型
def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    
    # CNN架构
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = build_model()
model.summary()

# 回调函数 <--
log_file = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
csv_logger = callbacks.CSVLogger(log_file)

checkpoint_path = os.path.join(checkpoint_dir, "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5")
model_checkpoint = callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1)

early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True)

# 尝试从检查点加载 <--
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Loading weights from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

# 训练模型
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[csv_logger, model_checkpoint, early_stopping])

# 保存最终模型 <--
final_model_path = os.path.join(checkpoint_dir, "final_model.h5")
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

# 评估模型
print("Evaluating model...")
loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy*100:.2f}%")