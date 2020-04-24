#---Import Dependencies---#

import numpy as np
import matplotlib.pyplot as plt
import os

from pickle import dump
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.models import Model

from sklearn.decomposition import PCA




#---Define Model and Feature Extractor---#

model = keras.applications.VGG19(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)




#---Load Dataset File Paths---#

images_path = 'dataset'
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]




#---Function Implementations---#

def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x




#---Extract Features---#

features = []
for image_path in tqdm(images):
    img, x = get_image(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)




#---Apply PCA---#
    
pca = PCA(n_components=500)
pca.fit(features)
pca_features = pca.transform(features)




#---Save Features and PCA Object---#
np.savetxt('pca_500_vgg19.out', pca_features)
dump(pca, open('pca_500.pkl', 'wb'))
