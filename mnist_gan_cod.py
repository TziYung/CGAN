#%%
#import module
import cv2
from tensorflow import keras
import pickle
import numpy
import time
from matplotlib import pyplot 
import random
import tensorflow as tf
from matplotlib.animation import FuncAnimation
import datetime
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# print(gpus)

#%%
# #import data
from tensorflow.keras.datasets.mnist import load_data
# from tensorflow.keras.datasets.fashion_mnist import load_data
(x_train, y_train), (_, _) = load_data()
x_train=numpy.expand_dims(x_train,axis= -1)
x_train=x_train.astype('float32')
x_train=(x_train-127.5)/127.5
real=[]
for index in range(x_train.shape[0]):
    real.append([[[y_train[index]],x_train[index]],1])
#%%

def create_gen():
    condition=keras.Input((1))
    layer=keras.layers.Embedding(10,64)(condition)
    layer=keras.layers.Dense(64)(layer)
    condition_out=keras.layers.Reshape((8,8,1))(layer)
    
    noise=keras.Input((100))
    layer=keras.layers.Dense(256)(noise)
    layer=keras.layers.BatchNormalization()(layer)
    activation=keras.layers.LeakyReLU()(layer)
    layer=keras.layers.Dense(4096)(activation)
    layer=keras.layers.BatchNormalization()(layer)
    activation=keras.layers.LeakyReLU()(layer)
    noise_out=keras.layers.Reshape((8,8,64))(activation)
    
    mix=keras.layers.Concatenate()([condition_out,noise_out])
    x=keras.layers.Conv2DTranspose(256,(4,4),strides=(2,2))(mix)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.LeakyReLU()(x)

    x=keras.layers.Conv2DTranspose(128,(4,4),strides=(2,2))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.LeakyReLU()(x)
    x=keras.layers.Conv2D(1,(11,11))(x)
    x=keras.layers.Activation('tanh')(x)
    
    generator=keras.Model([condition,noise],x)
    return generator
generator=create_gen()
generator.summary()

#%%
# Discriminator
def create_dis():
    condition=keras.Input((1))
    layer=keras.layers.Embedding(10,64)(condition)
    layer=keras.layers.Dense(784)(layer)
    condition_out=keras.layers.Reshape((28,28,1))(layer)
    img=keras.Input((28,28,1))
    x=keras.layers.Concatenate()([condition_out,img])
    x=keras.layers.Conv2D(128,(7,7))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.LeakyReLU()(x)
    x=keras.layers.Conv2D(128,(5,5))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.LeakyReLU()(x)
    x=keras.layers.Conv2D(64,(5,5))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.LeakyReLU()(x)
    # x=keras.layers.Conv2D(128,(5,5))(x)
    # x=keras.layers.BatchNormalization()(x)
    # x=keras.layers.LeakyReLU()(x)

    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(1)(x)
    x=keras.layers.Activation('sigmoid')(x)
    
    discriminator=keras.Model([condition,img],x)
    opt=tf.keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)
    return discriminator
discriminator=create_dis()
discriminator.summary()
#%%
#gan model
def define_model():
    global discriminator
    global generator
    discriminator.trainable=False
    label=keras.Input((1))
    noise=keras.Input((100))
    gan_input=[label,noise]
    gan_output=(discriminator([label,generator(gan_input)]))
    model=keras.Model(inputs=gan_input,outputs=gan_output)
    opt=tf.keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.summary()
    return model
model=define_model()
#%%

#%%
now=datetime.datetime.now()
# while True:
for x in range(60000):
    data=[]
    data.extend(random.choices(real,k=50))
    label=numpy.array([[n] for n in range(10)]*10)
    random.shuffle(label)
    noise=numpy.array([[random.uniform(-127.5,127.5)/127.5 for _ in range(100)] for _ in range(100)])
    gen=generator.predict([label,noise])
    for index in range(50): 
        data.append([[label[index],gen[index]],0])
    random.shuffle(data)
    x_label,x_img,y=[],[],[]
    for  index in range(len(data)):
        d=data[index][0]
        
        ans=data[index][1]
        x_label.append(d[0])
        x_img.append(d[1])
        y.append([ans])
    x_label=numpy.array(x_label)
    x_img=numpy.array(x_img)
    y=numpy.array(y)
    # train discriminator

    
    discriminator.train_on_batch([x_label,x_img],y)

    
    # train generator   
    model.train_on_batch([label,noise],numpy.array([1 for _ in range(label.shape[0])]))

    
    

    if x%600==0 and x!=0:
        print(x)
        loss,acc=discriminator.evaluate([x_label,x_img],y)
        loss,acc=model.evaluate([label,noise],numpy.array([1 for n in range(label.shape[0])]))
        print(datetime.datetime.now()-now)
        label=numpy.array([[n] for n in range(10)]*10)
        gen=generator.predict([label,noise[:100]])
        result=discriminator.predict([label,gen])
        pyplot.figure(dpi=1200)
        pyplot.title(x)
        for loop in range(10):
            for img in range(10):
                pyplot.subplot(10,10,img+1+loop*10)
                pyplot.xticks([])
                pyplot.yticks([])
                # pyplot.title(f"{result[img]},{label[img]}",fontsize=5)
                pyplot.imshow(gen[img+loop*10].reshape(28,28),cmap='gray_r')
        pyplot.savefig(f"D:\\mnist_gan\\{x}")
        pyplot.close('all')
        
        
        discriminator.save('discriminator.h5')
        generator.save('generator.h5')

    x+=1




