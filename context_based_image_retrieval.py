#---Import Dependencies---#

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from pickle import load
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tkinter import *
from tkinter import filedialog

import pyglet
from pyglet.window import Window, mouse, gl, key
from pyglet.gl import *

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.models import Model

#---Import Features and PCA Object---#

pca_features = np.loadtxt("pca_500_vgg19.out")
pca = load(open('pca_500.pkl', 'rb'))




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

def get_image_vector(query_image_path):
    img, x = get_image(query_image_path);
    features_image = feat_extractor.predict(x)[0]
    pca_features_image = pca.transform(features_image.reshape(1, -1))
    return pca_features_image

def get_closest_images(query_image_vector, num_results=5):
    distances = [distance.euclidean(query_image_vector, feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:num_results]
    return idx_closest

def assign_matched_images(path):
    matchlst =[]
    global match1, match2, match3, match4, match5
    
    query_image_vector = get_image_vector(path)
    idx_closest = get_closest_images(query_image_vector)

    for i in idx_closest:
        images[i].split('\\')[-1]
        matchlst.append(images[i].split('\\')[-1])
        
    image1 = pyglet.resource.image(matchlst[0])
    image1.anchor_x = image1.width//2
    image1.anchor_y = image1.height//2
    match1 = pyglet.sprite.Sprite(image1, 92 , 145)
    match1.scale = 0.4
    
    image2 = pyglet.resource.image(matchlst[1])
    image2.anchor_x = image2.width//2
    image2.anchor_y = image2.height//2
    match2 = pyglet.sprite.Sprite(image2, 264 , 145)
    match2.scale =0.4

    image3 = pyglet.resource.image(matchlst[2])
    image3.anchor_x = image3.width//2
    image3.anchor_y = image3.height//2
    match3 = pyglet.sprite.Sprite(image3, 448 , 145)
    match3.scale=0.4

    image4 = pyglet.resource.image(matchlst[3])
    image4.anchor_x = image4.width//2
    image4.anchor_y = image4.height//2
    match4 = pyglet.sprite.Sprite(image4, 627 , 145)
    match4.scale =0.4

    image5 = pyglet.resource.image(matchlst[4])
    image5.anchor_x = image5.width//2
    image5.anchor_y = image5.height//2
    match5 = pyglet.sprite.Sprite(image5, 811 , 145)
    match5.scale=0.4




#---Global Variables---#

bgimage= pyglet.resource.image('CBIR.png')
query =  0
match1 = 0
match2 = 0
match3 = 0
match4 = 0
match5 = 0
draw = False
drawm = False
file = ""
qpath = ""
qWidth = 200
dWidth = 150




#---Draw Pyglet Screen---#

glEnable(GL_TEXTURE_2D)
gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

display = pyglet.canvas.get_display()
screen = display.get_default_screen()
 
mycbir = pyglet.window.Window(907, 655,                     # setting window
              resizable=True,  
              caption="Content Based Image Retrieval",  
              config=pyglet.gl.Config(double_buffer=True),  # Avoids flickers
              vsync=False                                   # For flicker-free animation
              )                                             # Calling base class constructor
mycbir.set_location(screen.width // 2 - 300,screen.height//2 - 350)




#---Pyglet Event Functions---#

@mycbir.event
def on_draw():
    
    global query, qimage
    
    mycbir.clear()
    bgimage.blit(0,0)

    if draw == True:
        query.draw()
        
    if drawm == True:
        match1.draw()
        match2.draw()
        match3.draw()
        match4.draw()
        match5.draw()

@mycbir.event
def on_mouse_release(x, y, button, modifiers):

    global draw
    global drawm, query, qimage, file, qpath
   
    pyglet.resource.path = ['dataset']
    pyglet.resource.reindex()
    if button == mouse.LEFT:
        
        if (373 <= x <= 554) and (337 <= y <= 382):
            draw = False
            root = Tk()
            root.filename = filedialog.askopenfilename(filetypes =(("JPG files", "*.jpg"), ("All files", "*.*")))
            qpath = root.filename
            root.destroy()
            
            qpath.encode('unicode_escape')
            
            qimage = pyglet.image.load(qpath)
            qimage.anchor_x = qimage.width//2
            qimage.anchor_y = qimage.height//2
            
            query = pyglet.sprite.Sprite(qimage,455, 474)
            query.scale = qWidth / query.width 
            
            draw = True

        if (373 <= x <= 554) and (274 <= y <= 318):
            if draw== True:
                assign_matched_images(qpath)
                drawm = True
            
@mycbir.event  
def on_key_press(symbol, modifiers):
    if symbol == key.SPACE:
        print('z')
        mycbir.close()
    
def update(dt):
    dt




#---Run Application---#
    
pyglet.clock.schedule_interval(update, 1/20.)
pyglet.app.run()
