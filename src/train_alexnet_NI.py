import numpy as np
from keras.optimizers import SGD,Adagrad,RMSprop
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import save_model
from keras.models import load_model
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import TensorBoard,EarlyStopping,CSVLogger, ModelCheckpoint
from keras import applications
from keras.utils import plot_model
from keras.applications.vgg16 import preprocess_input
import sys
import os
import argparse
from alexnet import alexnet_model
# from image import ImageDataGenerator

naive=False  # 为true的话为cumulative模式
##############################
# Configuration settings
##############################
def mkdir(path):

    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path+' 创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-mn','--model_name',action='store',type=str,
        default = 'default',help='you must give a model name')
parser.add_argument('-dp','--data_path',action='store',type=str,
        default='../data',help='train and val  file path')
parser.add_argument('-lr','--learning_rate',action='store',type=float,
        default=0.01,help='learning_rate')
#parser.add_argument('-mt','--momentum',action='store',type=float,
#        default=0.9,help='learning_rate')
parser.add_argument('-ne','--num_epochs',action='store',type=int,
        default=10,help='num_epochs')
parser.add_argument('-bs','--batch_size',action='store',type=int,
        default=100,help='batch size')
parser.add_argument('-nc','--num_classes',action='store',type=int,
        default=2,help='num classes')   # no use now
parser.add_argument('-tl','--train_layers',nargs='+',action='store',type=str,
        default=['logit','linear'],help='layers need to be trained.')
# TODO
parser.add_argument('-tn','--top_N',action='store',type=int,
        default=5,help='whether the targets are in the top K predictions.')
parser.add_argument('-um','--use_model',action='store',type=str,
        default='',help='use model to initial.')
# TODO
parser.add_argument('-spe','--steps_per_epoch',action='store',type=int,
        default=100,help='train: steps_pre_epoch.')
parser.add_argument('-vs','--validation_steps',action='store',type=int,
        default=50,help='test: validation_steps.')


args = parser.parse_args()
print("="*50)
print("[INFO] args:\r")
print(args)
print("="*50)

train_data_dir = args.data_path + '/core50_128x128/s{}' #!!!!!!
validation_data_dir = args.data_path + '/core50_128x128/s2'


model_name = args.model_name

epochs = args.num_epochs

batch_size = args.batch_size

train_layers = args.train_layers

learning_rate  = args.learning_rate

use_model = args.use_model

steps_per_epoch = args.steps_per_epoch 

validation_steps = args.validation_steps 

S_PATH = sys.path[0]

DATA_PATH = args.data_path

TENSORBOARD_PATH = DATA_PATH + '/Graph/{}'.format(model_name)
mkdir(os.path.dirname(TENSORBOARD_PATH))
mkdir(TENSORBOARD_PATH)

LOG_PATH = DATA_PATH + '/log/training_{}.csv'.format(model_name)
mkdir(os.path.dirname(LOG_PATH))

BEST_WEIGHT = DATA_PATH + "/bestWeights/weight_{}.h5".format(model_name)
mkdir(os.path.dirname(BEST_WEIGHT))

END_WEIGHT = DATA_PATH + '/endWeights/weight_{}.h5'.format(model_name)
mkdir(os.path.dirname(END_WEIGHT))

END_MODEL = DATA_PATH + '/endModel/model_{}.h5'.format(model_name)
mkdir(os.path.dirname(END_MODEL))

img_width=128
img_height=128


if use_model == '' :
    print("*" * 50)
    print('[INFO] init train mode')
    print("*" * 50)

    # alexnet
    alex = alexnet_model(input_shape=(img_width,img_height,3),n_classes=50)

    model = Model(input=alex.input, output=alex.output)
    print(model.summary())

 # Adjusting the batch order
train_sess=[0, 1, 3, 4, 5, 7, 8, 10]
test_sess=[2, 6, 9]
app = [-1] * 8
batch_order=[x for x in range(8)]
for i, batch_idx in enumerate(batch_order):
        app[i] = train_sess[batch_idx]
train_sess = app
sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
# adadelta drop 0.5
model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                                    metrics=['accuracy'])


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    )



# tensor board
tbCallBack = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=0,  
                  write_graph=True, write_images=True)
#* tensorboard --logdir path_to_current_dir/Graph --port 8080 
print("tensorboard --logdir {} --port 8080".format(TENSORBOARD_PATH))


# earlystoping
# ES = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

# csv log
csvlog = CSVLogger(LOG_PATH,separator=',', append=True)

# saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint( monitor='val_acc',filepath=BEST_WEIGHT, verbose=1, save_best_only=True)

#################################
# fit
#################################
def gen_flow_for_two_inputs(generators):
        while True:
                for item in generators:
                        X1i=item.next()
                #Assert arrays are equal - this was for peace of mind, but slows down training
                #np.testing.assert_array_equal(X1i[0],X2i[0])
                        yield [X1i[0], X1i[1]]

if naive:
        for i in range(len(train_sess)):
                print("train session "+str(train_sess[i]))
                directories=train_data_dir.format(train_sess[i]+1)
                train_generator = train_datagen.flow_from_directory(directory=directories,
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')



                validation_generator = validation_datagen.flow_from_directory(directory=validation_data_dir,
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')
                steps_per_epoch=train_generator.n//train_generator.batch_size
                validation_steps=validation_generator.n//validation_generator.batch_size
                # begin to fit 
                model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=1,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[tbCallBack,csvlog,checkpointer],verbose=1)
        model.save_weights(END_WEIGHT)
        save_model(model,END_MODEL)                                                              
else:
        validation_generator = validation_datagen.flow_from_directory(directory=validation_data_dir,
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')
        file=open("labels.txt","w")
        label_map=validation_generator.class_indices
        label_map=dict((v,k) for k,v in label_map.items())
       
        print(label_map)
        for key,value in label_map.items():
                file.write(str(key)+" "+value+"\n")
        file.close()

        
        directories=[]
        for i in range(len(train_sess)):
                print("train session "+str(train_sess[i]))
                directories.append(train_data_dir.format(train_sess[i]+1))
                generator_lists=[]
                for directory in directories:
                        temp1 = train_datagen.flow_from_directory(directory=train_data_dir.format(train_sess[0]+1),
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
                        generator_lists.append(temp1)

                
                
                # print(labels)
                # print(classes)

                num=0
                for item in generator_lists:
                        num+=item.n
                steps_per_epoch=num //generator_lists[0].batch_size
                validation_steps=validation_generator.n//validation_generator.batch_size
                # begin to fit 
                model.fit_generator(gen_flow_for_two_inputs(generator_lists),
                    steps_per_epoch=steps_per_epoch,
                    epochs=1,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[tbCallBack,csvlog,checkpointer],verbose=1)
        model.save_weights(END_WEIGHT)
        save_model(model,END_MODEL)   

#################################
# model
#################################