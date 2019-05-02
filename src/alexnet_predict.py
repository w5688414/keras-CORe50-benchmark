
from keras.preprocessing import image
import numpy as np
from keras.models import Model, load_model 
from keras.applications.vgg16 import preprocess_input 

label_dict={}
with open("labels.txt","r") as f:
    for item in f.readlines():
        key,value=item.strip().split()
        label_dict[key]=value

model = load_model("/home/eric/data/CoRe50/bestWeights/weight_default.h5")

img_path = '/home/eric/data/CoRe50/core50_128x128/s1/o1/C_01_01_000.png'
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
preds=np.argmax(preds)
# print('Predicted:', decode_predictions(preds))
print(preds)
print(label_dict[str(preds)])