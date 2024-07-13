from scripts.data_preprocessing import load_and_preprocess_data, handle_text_data
from scripts.image_preprocessing import load_images, create_image_index_dict
from models.cnn_model import create_cnn_model
from models.rf_model import create_rf_model
from models.xgb_model import create_xgb_model
from models.dnn_model import create_dnn_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load and preprocess tabular data
pivoted_df, food_train_df = load_and_preprocess_data()
food_train_df = handle_text_data(food_train_df)

# Ensure all column names are strings
pivoted_df.columns = pivoted_df.columns.astype(str)

# Data split for tabular data
x_train, x_test, y_train, y_test = train_test_split(pivoted_df.drop('category', axis=1), pivoted_df['category'], test_size=0.2, random_state=42)

# Scale tabular data
scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

# Image preprocessing
images, labels = load_images()

# Convert labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# One-hot encode the labels for the CNN model
labels_cnn = to_categorical(labels, num_classes=len(label_encoder.classes_))

# Data split for image data
x_train_img, x_test_img, y_train_img, y_test_img = train_test_split(images, labels_cnn, test_size=0.2, random_state=42)

# Convert image data to TensorFlow datasets
train_dataset_img = tf.data.Dataset.from_tensor_slices((x_train_img, y_train_img)).batch(32)
val_dataset_img = tf.data.Dataset.from_tensor_slices((x_test_img, y_test_img)).batch(32)

# CNN Model
input_shape = (144, 144, 3)
num_classes = len(label_encoder.classes_)
cnn_model = create_cnn_model(input_shape, num_classes)
# Train and evaluate CNN model
cnn_model.fit(train_dataset_img, epochs=5, batch_size=32, verbose=1, validation_data=val_dataset_img)

# XGBoost Model
xgb_model = create_xgb_model()
xgb_model.fit(scaled_train, y_train)
print("XGBoost Model")
print(classification_report(y_train, xgb_model.predict(scaled_train)))
print(classification_report(y_test, xgb_model.predict(scaled_test)))

# Random Forest Model
rf_model = create_rf_model()
rf_model.fit(scaled_train, y_train)
print("Random Forest Model")
print(classification_report(y_train, rf_model.predict(scaled_train)))
print(classification_report(y_test, rf_model.predict(scaled_test)))

# DNN Model
input_shape_dnn = (scaled_train.shape[1],)
dnn_model = create_dnn_model(input_shape_dnn, num_classes)
# Convert tabular data to TensorFlow datasets for DNN model
train_dataset_dnn = tf.data.Dataset.from_tensor_slices((scaled_train, y_train)).batch(32)
val_dataset_dnn = tf.data.Dataset.from_tensor_slices((scaled_test, y_test)).batch(32)
dnn_model.fit(train_dataset_dnn, epochs=12, batch_size=32, verbose=1, validation_data=val_dataset_dnn)
