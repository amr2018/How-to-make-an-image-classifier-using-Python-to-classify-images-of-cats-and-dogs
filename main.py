# dataset https://www.kaggle.com/datasets/erkamk/cat-and-dog-images-dataset?resource=download&select=y.pickle
from sklearn.svm import LinearSVC
from PIL import Image
import numpy as np
import os
import joblib

X = []
y = []

model = LinearSVC(dual='auto')

# train our model on 50 images only.
def process_images(folder_name, class_name):
    for i in os.listdir(f'{folder_name}/{class_name}')[0:50]:
        image_path = f'{folder_name}/{class_name}/{i}'
        image = Image.open(image_path)
        image = image.resize((200, 200))
        image = np.array(image).flatten()
        X.append(image)
        y.append(class_name)

process_images('dataset', 'Cat')
process_images('dataset', 'Dog')

def train(model):
    model.fit(X, y)
    ## save the model
    joblib.dump(model, 'model.pickle')
    print('Model trained and saved')

def load_model():
    if not os.path.exists('model.pickle'):
        train(model)
        
        return joblib.load('model.pickle')
    else:
        return joblib.load('model.pickle')

model = load_model()

def predict(image_path):
    image = Image.open(image_path)
    image = image.resize((200, 200))
    image = np.array(image).flatten()
    print(model.predict([image]))

predict('dog.jpg')
