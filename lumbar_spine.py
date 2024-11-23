import os
import pandas as pd
import numpy as np
import random
import cv2
import pydicom
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPUs.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
else:
    print("No GPU detected, training on CPU.")

BASE_PATH = 'C:/Users/GOKUL AKASH S/Downloads/rsna-2024-lumbar-spine-degenerative-classification'
IMAGES_PATH = os.path.join(BASE_PATH, 'train_images')
TEST_IMAGES_PATH = os.path.join(BASE_PATH, 'test_image')
CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
IMG_SIZE = 128
NUM_SAMPLES = 1000  
SEED = 42

labels_df = pd.read_csv(CSV_PATH)

def gather_dcm_paths(images_path):
    dcm_files = []
    for root, _, files in os.walk(images_path):
        for file in files:
            if file.endswith('.dcm'):
                dcm_files.append(os.path.join(root, file))
    return dcm_files

all_dcm_files = gather_dcm_paths(IMAGES_PATH)
random.seed(SEED)
subset_dcm_files = random.sample(all_dcm_files, min(NUM_SAMPLES, len(all_dcm_files)))

def preprocess_dicom(file_path, img_size=IMG_SIZE):
    try:
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_resized = cv2.resize(img, (img_size, img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        return img_rgb
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

images = []
file_study_mapping = []
for file_path in tqdm(subset_dcm_files, desc="Preprocessing DICOM files"):
    img = preprocess_dicom(file_path)
    if img is not None:
        images.append(img)
        study_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        file_study_mapping.append((file_path, study_id))

images = np.array(images, dtype=np.float32)
print(f"Total valid images loaded: {len(images)}")

file_study_df = pd.DataFrame(file_study_mapping, columns=['file_path', 'study_id'])
labels_df['study_id'] = labels_df['study_id'].astype(str)
file_study_df['study_id'] = file_study_df['study_id'].astype(str)
merged_df = pd.merge(file_study_df, labels_df, on='study_id', how='inner')

merged_df['spinal_canal_stenosis_l1_l2'] = merged_df['spinal_canal_stenosis_l1_l2'].fillna('Unknown').astype(str)

label_encoder = LabelEncoder()
merged_df['label'] = label_encoder.fit_transform(merged_df['spinal_canal_stenosis_l1_l2'])
labels = merged_df['label'].values

images = images[:len(labels)]  

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, random_state=SEED)

def create_dataset(X, y, batch_size=16):
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

batch_size = 4
train_dataset = create_dataset(X_train, y_train, batch_size)
val_dataset = create_dataset(X_val, y_val, batch_size)

def create_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(label_encoder.classes_)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

def create_resnet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(label_encoder.classes_)):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
    return model

def create_efficientnet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(label_encoder.classes_)):
    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

def train_and_evaluate_model(model):
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=4,
                        verbose=0) 
                        
    loss, accuracy = model.evaluate(val_dataset) 
    return accuracy

cnn_model = create_cnn()
resnet_model = create_resnet()
efficientnet_model = create_efficientnet()

cnn_acc = train_and_evaluate_model(cnn_model)
resnet_acc = train_and_evaluate_model(resnet_model)
effnet_acc = train_and_evaluate_model(efficientnet_model)

test_dcm_files = gather_dcm_paths(TEST_IMAGES_PATH)

test_images_list=[]
for file_path in tqdm(test_dcm_files):
    img=preprocess_dicom(file_path) 
    if img is not None:
        test_images_list.append(img)

test_images=np.array(test_images_list,dtype=np.float32)

def predict_and_save_results(model,test_images):
    preds=model.predict(test_images) 
    preds_labels=np.argmax(preds ,axis=1) 
   
    results=pd.DataFrame({
       'file_path': test_dcm_files,
       'predicted_label': preds_labels,
       'predicted_class': label_encoder.inverse_transform(preds_labels)  
   })
   
    return results

cnn_results=predict_and_save_results(cnn_model,test_images) 
resnet_results=predict_and_save_results(resnet_model,test_images) 
effnet_results=predict_and_save_results(efficientnet_model,test_images)

results_summary=pd.DataFrame({
   'Model': ['CNN', 'ResNet', 'EfficientNet'],
   'Validation Accuracy (%)': [cnn_acc * 100,resnet_acc * 100 ,effnet_acc * 100]
})

print(results_summary)

plt.figure(figsize=(10 ,5))
plt.bar(results_summary['Model'], results_summary['Validation Accuracy (%)'], color=['blue', 'orange', 'green'])
plt.title('Model Validation Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.show()

def display_classified_images(results):
   plt.figure(figsize=(15 ,15))
   selected_indices_1s = results[results["predicted_label"] == 1].index.tolist()
   selected_indices_0s = results[results["predicted_label"] == 0].index.tolist()

   num_to_display_per_class_1s=min(4,len(selected_indices_1s))
   num_to_display_per_class_0s=min(4,len(selected_indices_0s))

   selected_indices_to_plot=[]
   selected_indices_to_plot.extend(random.sample(selected_indices_1s[:num_to_display_per_class_1s], num_to_display_per_class_1s)) 
   selected_indices_to_plot.extend(random.sample(selected_indices_0s[:num_to_display_per_class_0s], num_to_display_per_class_0s)) 

   for i in selected_indices_to_plot:
       img=cv2.cvtColor(test_images[i].astype(np.uint8), cv2.COLOR_RGB2BGR) 
       label="Severe Stenosis" if results["predicted_label"].iloc[i] == 1 else "No Severe Stenosis"
       
       if results["predicted_label"].iloc[i] == 1:
           coords_x_start=20  
           coords_y_start=20  
           coords_x_end=100  
           coords_y_end=100  
           cv2.rectangle(img,(coords_x_start ,coords_y_start),(coords_x_end ,coords_y_end),(255 ,0 ,0),3)
           cv2.putText(img,label,(coords_x_start ,coords_y_start - 10),cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255 ,255 ,255),1)

       plt.subplot(4 ,4 ,selected_indices_to_plot.index(i) + 1) 
       plt.imshow(img) 
       plt.title(label) 
       plt.axis('off') 
   plt.tight_layout()
   plt.show()

display_classified_images(cnn_results)