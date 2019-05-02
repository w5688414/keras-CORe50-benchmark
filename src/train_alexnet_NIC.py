import numpy as np
from keras.optimizers import SGD,Adagrad,RMSprop
from keras import regularizers
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import save_model
from keras.models import load_model
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import TensorBoard,EarlyStopping,CSVLogger, ModelCheckpoint
from keras import applications
from keras.utils import plot_model
from keras.applications.vgg16 import preprocess_input
from alexnet_CWR import alexnet_cwr_model
import sys
import os
import argparse
from alexnet import alexnet_model
from keras.initializers import normal
import pandas as pd #引入pandas
# from image import ImageDataGenerator

naive=False  
cumulative=False
CWR=True
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

LOG_PATH = DATA_PATH + '/log/training_alexnet_{}.csv'.format(model_name)
mkdir(os.path.dirname(LOG_PATH))

BEST_WEIGHT = DATA_PATH + "/bestWeights/weight_{}.h5".format(model_name)
mkdir(os.path.dirname(BEST_WEIGHT))

END_WEIGHT = DATA_PATH + '/endWeights/weight_{}.h5'.format(model_name)
mkdir(os.path.dirname(END_WEIGHT))

END_MODEL = DATA_PATH + '/endModel/model_{}.h5'.format(model_name)
mkdir(os.path.dirname(END_MODEL))

img_width=128
img_height=128

# weight initialization
cw=False
tw=True
if(cw):
        kernel_initializer='zero'
elif(tw):
        kernel_initializer=normal(mean=0.0, stddev=0.01, seed=None)
else:
        kernel_initializer='uniform'

if use_model == '' :
    print("*" * 50)
    print('[INFO] init train mode')
    print("*" * 50)

    # alexnet
    if(CWR):
            alex=alexnet_cwr_model(input_shape=(img_width,img_height,3),n_classes=50,kernel_initializer=kernel_initializer)
    else:
            alex = alexnet_model(input_shape=(img_width,img_height,3),n_classes=50)

    model = Model(input=alex.input, output=alex.output)
    print(model.summary())

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
                                   fill_mode='nearest'
                                   )

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
root_dir='/home/eric/Documents/Experiments/core50/extras/batches_filelists/NIC_inc/run0'
filename='train_batch_00_filelist.txt'
filepath=os.path.join(root_dir,filename)

def getDataFrame(filepath):
        traindf=pd.read_csv(filepath,sep=' ',header=None) #加载papa.txt,指定它的分隔符是 \t
        traindf.rename(columns={0:"filename",1:'class'},inplace=True)
        traindf['class']=traindf['class'].astype("str")
        return traindf
directory='/home/eric/data/CoRe50/core50_128x128'


testfilename='test_filelist.txt'
testfilepath=os.path.join(root_dir,testfilename)
validdf=pd.read_csv(testfilepath,sep=' ',header=None) #加载papa.txt,指定它的分隔符是 \t
validdf.rename(columns={0:"filename",1:'class'},inplace=True)
validdf['class']=validdf['class'].astype("str")

labels=[]
for i in range(50):
        labels.append(str(i))

validation_generator = validation_datagen.flow_from_dataframe(
                                                              dataframe=validdf,
                                                              directory=directory,
                                                              x_col="filename",
                                                              y_col="class",
                                                              subset="training",
                                                              classes=labels,
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

def copyModel2Model(pre_model,model,_classes,class_list):
        pre_weights_list=pre_model.get_layer('mid_fc8').get_weights()
        pre_weight=pre_weights_list[0]
        pre_bias=pre_weights_list[1]
        cur_weight_list=model.get_layer('mid_fc8').get_weights()
        cur_weight=cur_weight_list[0]
        cur_bias=cur_weight_list[1]
        te_new_weights = []
        # te_new_biases = []
        for item in _classes:
                index=int(item)
                if(item in class_list):
                        # print(pre_weight.shape)
                        # print(pre_weight[:,index].shape)
                        te_new_weights.append(pre_weight[:,index])
                        te_new_weights.append(cur_weight[:,index])
                        cur_weight[:,index]=np.mean(te_new_weights)
                        cur_bias[index]=(cur_bias[index]+pre_bias[index])/2
                        print("mid_fc8 {} weights updated ".format(item))
                        # print(cur_weight.shape)
                        # print(cur_bias.shape)
                else:
                        cur_weight[index]=pre_weight[index]
                        cur_bias[index]=pre_bias[index]
        cur_weight_list[0]=cur_weight
        cur_weight_list[1]=cur_bias
        model.get_layer('mid_fc8').set_weights(cur_weight_list)
        return model





if naive:
        for i in range(79):
                batch_count='%02d' % i
                filename='train_batch_{}_filelist.txt'.format(batch_count)
                print("load dataset: "+filename)
                filepath=os.path.join(root_dir,filename)
                traindf=getDataFrame(filepath)
                train_generator = train_datagen.flow_from_dataframe(
                                                    dataframe=traindf,
                                                    directory=directory,
                                                    x_col="filename",
                                                    y_col="class",
                                                    subset="training",
                                                    classes=labels,
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
                validation_generator = validation_datagen.flow_from_dataframe(
                                                              dataframe=validdf,
                                                              directory=directory,
                                                              x_col="filename",
                                                              y_col="class",
                                                              subset="training",
                                                              classes=labels,
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
elif(cumulative):
        
        file=open("labels.txt","w")
        label_map=validation_generator.class_indices
        label_map=dict((v,k) for k,v in label_map.items())
       
        print(label_map)
        for key,value in label_map.items():
                file.write(str(key)+" "+value+"\n")
        file.close()

        
        dfs=[]
        for i in range(79):
                batch_count='%02d' % i
                filename='train_batch_{}_filelist.txt'.format(batch_count)
                print("load dataset: "+filename)
                filepath=os.path.join(root_dir,filename)
                traindf=getDataFrame(filepath)
                dfs.append(traindf)
                generator_lists=[]
                for df in dfs:
                        temp1 = train_datagen.flow_from_dataframe(
                                                    dataframe=df,
                                                    directory=directory,
                                                    x_col="filename",
                                                    y_col="class",
                                                    subset="training",
                                                    classes=labels,
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
elif(CWR):
        pre_model=None
        class_list=[]
        for i in range(79):
                batch_count='%02d' % i
                filename='train_batch_{}_filelist.txt'.format(batch_count)
                print("load dataset: "+filename)
                filepath=os.path.join(root_dir,filename)
                traindf=getDataFrame(filepath)
                train_generator = train_datagen.flow_from_dataframe(
                                                    dataframe=traindf,
                                                    directory=directory,
                                                    x_col="filename",
                                                    y_col="class",
                                                    subset="training",
                                                    classes=labels,
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
                steps_per_epoch=train_generator.n//train_generator.batch_size
                validation_steps=validation_generator.n//validation_generator.batch_size
                _classes=traindf['class'].unique()
                
                if(i==1):
                        pre_model=model
                if(i>1):
                        model=copyModel2Model(pre_model,model,_classes,class_list)
                        
                print(_classes)
                print(class_list)
                class_list.extend(_classes)
                
               
                model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=1,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=[tbCallBack,csvlog,checkpointer],verbose=1)

model.save_weights(END_WEIGHT)
save_model(model,END_MODEL)   