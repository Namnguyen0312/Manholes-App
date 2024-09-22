import os
import pandas as pd
import numpy as np
from keras import Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16
from keras.src.applications.densenet import DenseNet121
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array, to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


# Bước 1: Load dữ liệu huấn luyện từ thư mục
def load_training_data(data_dir, image_size=(224, 224)):
    categories = ['dourec', 'rec', 'roundsqr', 'sqr']
    data = []
    labels = []

    for category in categories:
        folder_path = os.path.join(data_dir, category)
        class_idx = categories.index(category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img) / 255.0  # Chuẩn hóa hình ảnh
            data.append(img)
            labels.append(class_idx)

    return np.array(data), np.array(labels), categories


# Tạo mô hình MobileNetV2
def create_model(num_classes, input_shape=(224, 224, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze các layer của MobileNetV2

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Tạo mô hình VGG16
def create_vgg16_model(num_classes, input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze các layer của VGG16

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Tạo mô hình ResNet50
def create_resnet50_model(num_classes, input_shape=(224, 224, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze các layer của ResNet50

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Tạo mô hình DenseNet121
def create_densenet121_model(num_classes, input_shape=(224, 224, 3)):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze các layer của DenseNet121

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Tạo mô hình InceptionV3
def create_inceptionv3_model(num_classes, input_shape=(224, 224, 3)):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze các layer của InceptionV3

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Tạo mô hình EfficientNetB0
def create_efficientnetb0_model(num_classes, input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze các layer của EfficientNetB0

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Bước 3: Huấn luyện mô hình và lưu mô hình
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10, model_path='saved_model.h5'):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Huấn luyện mô hình
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs)

    # Lưu mô hình sau khi huấn luyện
    model.save(model_path)
    print(f"Mô hình đã được lưu tại {model_path}")

    return history  # Trả về lịch sử huấn luyện


# Bước 4: Huấn luyện và đánh giá các mô hình
def train_and_evaluate_models(X_train, y_train, X_val, y_val, categories, model_paths, batch_size=32, epochs=10):
    # Tạo các mô hình
    models = {
        'MobileNetV2': create_model(num_classes=len(categories)),
        'VGG16': create_vgg16_model(num_classes=len(categories)),
        'ResNet50': create_resnet50_model(num_classes=len(categories)),
        'DenseNet121': create_densenet121_model(num_classes=len(categories)),
        'InceptionV3': create_inceptionv3_model(num_classes=len(categories)),
        'EfficientNetB0': create_efficientnetb0_model(num_classes=len(categories))
    }

    results = []  # Danh sách để lưu kết quả tổng quan của các mô hình
    epoch_details = []  # Danh sách để lưu chi tiết từng epoch

    # Huấn luyện từng mô hình và lưu chúng
    for model_name, model in models.items():
        print(f"Training {model_name} model...")
        model_path = model_paths[model_name]
        history = train_model(model, X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=epochs,
                              model_path=model_path)
        print(f"{model_name} model saved to {model_path}")

        # Đánh giá hiệu năng trên tập validation
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        print(f"{model_name} - Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

        # Lưu kết quả tổng quan của mô hình
        results.append({
            'Model': model_name,
            'Validation Accuracy': val_accuracy,
            'Validation Loss': val_loss,
            'Epochs': epochs,
            'Batch Size': batch_size,
            'Training Samples': len(X_train),
            'Validation Samples': len(X_val)
        })

        # Lưu chi tiết của từng epoch
        for epoch in range(epochs):
            epoch_details.append({
                'Model': model_name,
                'Epoch': epoch + 1,
                'Training Loss': history.history['loss'][epoch],
                'Training Accuracy': history.history['accuracy'][epoch],
                'Validation Loss': history.history['val_loss'][epoch],
                'Validation Accuracy': history.history['val_accuracy'][epoch],
                'Batch Size': batch_size
            })

    # Trả về kết quả
    return models, results, epoch_details


# Bước 5: Load mô hình đã lưu và phân loại
def classify_images_from_excel(model_path, excel_path, categories, output_excel_path):
    model = load_model(model_path)  # Tải mô hình đã lưu
    df = pd.read_excel(excel_path)
    df['Image_Path'] = df['Folder'].astype(str) + '/' + df['Filename'].astype(str)

    predictions = []
    for img_path in df['Image_Path']:
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img) / 255.0  # Chuẩn hóa hình ảnh
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)
            predicted_class = np.argmax(pred, axis=1)[0]
            predictions.append(categories[predicted_class])
        else:
            predictions.append('Unknown')  # Nếu hình ảnh không tồn tại

    df['Class'] = predictions
    df.to_excel(output_excel_path, index=False)
    print(f"Kết quả phân loại đã được lưu tại {output_excel_path}")


# Pipeline chính
def main():
    # Bước 1: Chuẩn bị dữ liệu huấn luyện
    data_dir = r'D:\Github\Manholes-App\Classification'
    excel_file = r'D:\Github\Manholes-App\1.xlsx'
    output_excel = r'D:\Github\Manholes-App\2.xlsx'

    # Đường dẫn lưu các mô hình
    model_paths = {
        'MobileNetV2': 'saved_model_mobilenetv2.h5',
        'VGG16': 'saved_model_vgg16.h5',
        'ResNet50': 'saved_model_resnet50.h5',
        'DenseNet121': 'saved_model_densenet121.h5',
        'InceptionV3': 'saved_model_inceptionv3.h5',
        'EfficientNetB0': 'saved_model_efficientnetb0.h5'
    }

    # Bước 2: Load dữ liệu huấn luyện
    X, y, categories = load_training_data(data_dir)
    #
    # # Bước 3: Chuẩn bị dữ liệu cho huấn luyện
    # y = to_categorical(y, num_classes=len(categories))
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # Các siêu tham số cho huấn luyện
    # batch_size = 32
    # epochs = 10
    #
    # # Bước 4: Huấn luyện và so sánh các mô hình
    # trained_models, results, epoch_details = train_and_evaluate_models(X_train, y_train, X_val, y_val, categories,
    #                                                                    model_paths, batch_size=batch_size,
    #                                                                    epochs=epochs)
    #
    # # Bước 5: Lưu kết quả tổng quan vào file Excel
    # df_results = pd.DataFrame(results)
    # comparison_excel = 'model_comparison_results.xlsx'
    # df_results.to_excel(comparison_excel, index=False)
    # print(f"Kết quả so sánh các mô hình đã được lưu tại {comparison_excel}")
    #
    # # Lưu chi tiết của từng epoch vào file Excel khác
    # df_epoch_details = pd.DataFrame(epoch_details)
    # epoch_details_excel = 'model_epoch_details.xlsx'
    # df_epoch_details.to_excel(epoch_details_excel, index=False)
    # print(f"Chi tiết huấn luyện từng epoch đã được lưu tại {epoch_details_excel}")

    # Bước 6: Sử dụng một mô hình đã lưu (ví dụ: MobileNetV2) để phân loại hình ảnh từ file Excel
    classify_images_from_excel(model_paths['InceptionV3'], excel_file, categories, output_excel)


if __name__ == '__main__':
    main()
