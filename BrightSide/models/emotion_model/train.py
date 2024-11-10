from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import model_from_json # type: ignore
import numpy as np
import pandas as pd
from model import create_model

def preprocess_data(df):
    X_train, train_y, X_test, test_y = [], [], [], []

    for index, row in df.iterrows():
        val = row["pixels"].split(" ")
        try:
            if "Training" in row["Usage"]:
                X_train.append(np.array(val, "float32"))
                train_y.append(row["emotion"])
            elif "PublicTest" in row["Usage"]:
                X_test.append(np.array(val, "float32"))
                test_y.append(row["emotion"])
        except:
            print(f"Error occurred at index: {index} and row: {row}")

    num_labels = 7
    X_train = np.array(X_train, "float32")
    train_y = np.array(train_y, "float32")
    X_test = np.array(X_test, "float32")
    test_y = np.array(test_y, "float32")

    train_y = to_categorical(train_y, num_classes=num_labels)
    test_y = to_categorical(test_y, num_classes=num_labels)

    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)
    X_test -= np.mean(X_test, axis=0)
    X_test /= np.std(X_test, axis=0)
    
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

    return X_train, train_y, X_test, test_y

def train_model(X_train, train_y, X_test, test_y, batch_size=64, epochs=30):
    model = create_model(X_train.shape[1:], num_labels=7)
    train_generator = ImageDataGenerator().flow(X_train, train_y, batch_size=batch_size)
    
    model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs
    )

    model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, validation_data=(X_test, test_y), shuffle=True)
    
    return model

def evaluate_model(model, X_train, train_y, X_test, test_y):
    train_score = model.evaluate(X_train, train_y, verbose=0)
    print("Train loss:", train_score[0])
    print("Train accuracy:", 100 * train_score[1])
    
    test_score = model.evaluate(X_test, test_y, verbose=0)
    print("Test loss:", test_score[0])
    print("Test accuracy:", 100 * test_score[1])

def save_model(model):
    fer_json = model.to_json()
    with open("fer3.json", "w") as json_file:
        json_file.write(fer_json)
    model.save_weights("fer3.weights.h5")
