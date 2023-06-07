'\nWe want to retrain a new model that takes\n1. raw image and 2. the expected text as input and outputs a boolean indicating whether the image is tampered\n\ntraining data:\ntrue: rendering variations\nfalse: 1. different words 2. perturbations\n\n'
import math,time,os,pathlib,cv2,numpy as np
from tensorflow.keras.datasets import mnist
from emnist import extract_training_samples,extract_test_samples
import re,sys
gtpu=False
import random,tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input,Dropout
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import code
from tensorflow.python.keras.utils.layer_utils import filter_empty_layer_containers
from cleverhans.tf2.utils import get_or_guess_labels,set_with_mask
if gtpu:sys.path.append('/usr/share/models');from official.common import distribute_utils;from official.utils.flags import core as flags_core;from official.utils.misc import model_helpers;from official.vision.image_classification.resnet import common
class CNN_EMNIST:
	@staticmethod
	def mod1(categories,input_shape=(32,32,1)):model=Sequential();model.add(Conv2D(32,(5,5),input_shape=input_shape,activation='relu',padding='same'));model.add(Conv2D(32,(5,5),activation='relu',padding='same'));model.add(MaxPooling2D(pool_size=(2,2)));model.add(Conv2D(128,(3,3),activation='relu',padding='same'));model.add(Conv2D(128,(3,3),activation='relu',padding='same'));model.add(MaxPooling2D(pool_size=(2,2)));model.add(Conv2D(256,(3,3),activation='relu',padding='same'));model.add(Conv2D(256,(3,3),activation='relu',padding='same'));model.add(MaxPooling2D(pool_size=(2,2)));model.add(Flatten());model.add(Dense(64));model.add(Activation('relu'));model.add(Dropout(.2));model.add(Dense(categories));model.add(Activation('softmax'));return model
	@staticmethod
	def mod1_tf2(categories,input_shape=(32,32,1)):inputs=Input(shape=input_shape);x=Conv2D(32,(5,5),input_shape=input_shape,padding='same')(inputs);x=Activation('relu')(x);x=Conv2D(32,(5,5),activation='relu',padding='same')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Conv2D(128,(3,3),activation='relu',padding='same')(x);x=Conv2D(128,(3,3),activation='relu',padding='same')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Conv2D(256,(3,3),activation='relu',padding='same')(x);x=Conv2D(256,(3,3),activation='relu',padding='same')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Flatten()(x);x=Dense(128,activation='relu')(x);exp_char=Input(shape=(categories,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([inputs,exp_char],x,name='mod1_tf2');return model
	@staticmethod
	def mod2(categories,input_shape=(32,32,1)):ly1_neurons=128;ly2_neurons=128;model=Sequential();print(input_shape);model.add(Conv2D(128,(3,3),input_shape=input_shape,activation='relu'));model.add(MaxPooling2D(pool_size=(2,2)));model.add(Dropout(.2));model.add(Conv2D(ly2_neurons,(3,3),activation='relu'));model.add(MaxPooling2D(pool_size=(2,2)));model.add(Dropout(.2));model.add(Flatten());model.add(Dense(categories));model.add(Activation('sigmoid'));model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']);return model
	@staticmethod
	def mod2_tf2(categories,input_shape=(32,32,1)):ly1_neurons=128;ly2_neurons=128;inputs=Input(shape=input_shape);x=Conv2D(128,(3,3),padding='same')(inputs);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Conv2D(ly2_neurons,(3,3),activation='relu')(x);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Flatten()(x);exp_char=Input(shape=(categories,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([inputs,exp_char],x,name='mod2_tf2');return model
	@staticmethod
	def mod3_tf2(categories,input_shape=(32,32,1)):ly1_neurons=64;ly2_neurons=64;inputs=Input(shape=input_shape);x=Conv2D(ly1_neurons,(3,3),padding='same')(inputs);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Conv2D(ly2_neurons,(3,3),activation='relu')(x);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Flatten()(x);exp_char=Input(shape=(categories,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([inputs,exp_char],x,name='mod2_tf2');return model
	@staticmethod
	def mod4_tf2(categories,input_shape=(32,32,1)):ly1_neurons=32;ly2_neurons=32;inputs=Input(shape=input_shape);x=Conv2D(ly1_neurons,(3,3),padding='same')(inputs);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Conv2D(ly2_neurons,(3,3),activation='relu')(x);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Flatten()(x);exp_char=Input(shape=(categories,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([inputs,exp_char],x,name='mod2_tf2');return model
	@staticmethod
	def mod5_tf2(categories,input_shape=(32,32,1)):ly1_neurons=16;ly2_neurons=16;inputs=Input(shape=input_shape);x=Conv2D(ly1_neurons,(3,3),padding='same')(inputs);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Conv2D(ly2_neurons,(3,3),activation='relu')(x);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Flatten()(x);exp_char=Input(shape=(categories,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([inputs,exp_char],x,name='mod2_tf2');return model
	@staticmethod
	def mod6_tf2(categories,input_shape=(32,32,1)):ly1_neurons=8;ly2_neurons=8;inputs=Input(shape=input_shape);x=Conv2D(ly1_neurons,(3,3),padding='same')(inputs);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Conv2D(ly2_neurons,(3,3),activation='relu')(x);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Flatten()(x);exp_char=Input(shape=(categories,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([inputs,exp_char],x,name='mod2_tf2');return model
	@staticmethod
	def mod7_tf2(categories,input_shape=(32,32,1)):ly1_neurons=4;ly2_neurons=4;inputs=Input(shape=input_shape);x=Conv2D(ly1_neurons,(3,3),padding='same')(inputs);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Conv2D(ly2_neurons,(3,3),activation='relu')(x);x=Activation('relu')(x);x=MaxPooling2D(pool_size=(2,2))(x);x=Dropout(.2)(x);x=Flatten()(x);exp_char=Input(shape=(categories,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([inputs,exp_char],x,name='mod2_tf2');return model
def same(a,b):difference=cv2.subtract(a,b);return not np.any(difference)
class ResNet:
	@staticmethod
	def residual_module(data,K,stride,chanDim,red=False,reg=.0001,bnEps=2e-05,bnMom=.9):
		shortcut=data;bn1=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(data);act1=Activation('relu')(bn1);conv1=Conv2D(int(K*.25),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1);bn2=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1);act2=Activation('relu')(bn2);conv2=Conv2D(int(K*.25),(3,3),strides=stride,padding='same',use_bias=False,kernel_regularizer=l2(reg))(act2);bn3=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv2);act3=Activation('relu')(bn3);conv3=Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)
		if red:shortcut=Conv2D(K,(1,1),strides=stride,use_bias=False,kernel_regularizer=l2(reg))(act1)
		x=add([conv3,shortcut]);return x
	@staticmethod
	def build(width,height,depth,classes,stages,filters,reg=.0001,bnEps=2e-05,bnMom=.9,dataset='cifar'):
		inputShape=height,width,depth;chanDim=-1
		if K.image_data_format()=='channels_first':inputShape=depth,height,width;chanDim=1
		inputs=Input(shape=inputShape);x=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(inputs);x=Conv2D(filters[0],(3,3),use_bias=False,padding='same',kernel_regularizer=l2(reg))(x)
		for i in range(0,len(stages)):
			stride=(1,1)if i==0 else(2,2);x=ResNet.residual_module(x,filters[i+1],stride,chanDim,red=True,bnEps=bnEps,bnMom=bnMom)
			for j in range(0,stages[i]-1):x=ResNet.residual_module(x,filters[i+1],(1,1),chanDim,bnEps=bnEps,bnMom=bnMom)
		x=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x);x=Activation('relu')(x);x=AveragePooling2D((8,8))(x);x=Flatten()(x);x=Dense(classes,kernel_regularizer=l2(reg))(x);x=Activation('softmax')(x);model=Model(inputs,x,name='resnet');return model
class RV:
	def __init__():0
	@staticmethod
	def residual_module(data,K,stride,chanDim,red=False,reg=.0001,bnEps=2e-05,bnMom=.9):
		shortcut=data;bn1=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(data);act1=Activation('relu')(bn1);conv1=Conv2D(int(K*.25),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1);bn2=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1);act2=Activation('relu')(bn2);conv2=Conv2D(int(K*.25),(3,3),strides=stride,padding='same',use_bias=False,kernel_regularizer=l2(reg))(act2);bn3=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv2);act3=Activation('relu')(bn3);conv3=Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)
		if red:shortcut=Conv2D(K,(1,1),strides=stride,use_bias=False,kernel_regularizer=l2(reg))(act1)
		x=add([conv3,shortcut]);return x
	@staticmethod
	def build(width,height,depth,classes,stages,filters,reg=.0001,bnEps=2e-05,bnMom=.9,dataset='cifar'):
		inputShape=height,width,depth;chanDim=-1
		if K.image_data_format()=='channels_first':inputShape=depth,height,width;chanDim=1
		inputs=Input(shape=inputShape);x=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(inputs);x=Conv2D(filters[0],(3,3),use_bias=False,padding='same',kernel_regularizer=l2(reg))(x)
		for i in range(0,len(stages)):
			stride=(1,1)if i==0 else(2,2);x=ResNet.residual_module(x,filters[i+1],stride,chanDim,red=True,bnEps=bnEps,bnMom=bnMom)
			for j in range(0,stages[i]-1):x=ResNet.residual_module(x,filters[i+1],(1,1),chanDim,bnEps=bnEps,bnMom=bnMom)
		x=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x);x=Activation('relu')(x);x=AveragePooling2D((8,8))(x);x=Flatten()(x);exp_char=Input(shape=(classes,));x=keras.layers.concatenate([x,exp_char]);x=Dense(32,activation='sigmoid',kernel_regularizer=l2(reg))(x);x=Dense(1,activation='sigmoid',kernel_regularizer=l2(reg))(x);model=Model(inputs=[inputs,exp_char],outputs=x,name='resnet');return model
class OurData:
	def __init__(self):
		self.emnist_label_names='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt';self.emnist_by_class_names='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';self.single_char_names='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`!@#$%^&*()-=~_+[]\\{}|;\':",./<>?';self.equal_chars='cijklmopsuvwxyz';self.equal_chars_map={}
		for char in self.equal_chars:self.equal_chars_map[char.upper()]=char
		self.equal_chars_map['`']="'";self.equal_chars_map['|']='/';self.valid_out=[x for x in self.single_char_names if x not in self.equal_chars_map]
		for char in self.single_char_names:
			if char not in self.equal_chars_map:self.equal_chars_map[char]=char
	def load_test_dataset(self,label_collapse=False):
		pattern2=re.compile('(\\d+)_(\\d+).png');data=[];exp=[];labels=[];path='text_baseline_data/single_char/test/windows_true';files=[x for x in os.listdir(path)if x.endswith('png')]
		for fname in files:img=cv2.imread(os.path.join(path,fname),cv2.IMREAD_GRAYSCALE);img=cv2.bitwise_not(img);mat=re.match(pattern2,fname);assert mat;char,font=int(mat.group(1)),mat.group(2);assert char<len(self.single_char_names)and char>-1;data.append(img);exp.append(char)
		labels+=[1]*len(exp);path='text_baseline_data/single_char/test/windows_false';files=[x for x in os.listdir(path)if x.endswith('png')]
		for fname in files:img=cv2.imread(os.path.join(path,fname),cv2.IMREAD_GRAYSCALE);img=cv2.bitwise_not(img);mat=re.match(pattern2,fname);assert mat;char,font=int(mat.group(1)),mat.group(2);assert char<len(self.single_char_names)and char>-1;data.append(img);exp.append(char)
		labels+=[0]*len(files)
		if label_collapse:
			exp=np.array(exp)
			for i in range(len(self.single_char_names)):exp[exp==i]=self.valid_out.index(self.equal_chars_map[self.single_char_names[i]])
			assert len(np.unique(exp))<=len(self.valid_out),np.unique(exp)
		for i in range(len(data)):cv2.imwrite('text_baseline_data/single_char/test/processed/{}_{}_{}.png'.format(labels[i],exp[i],i),data[i])
		return data,exp,labels
	def load_train_failed(self,get_false=False,path='text_baseline_data/continue_training',label_collapse=False,get_new_data=False,get_variants=True):
		save_path='text_baseline_data/continue_training_save.npz'if not label_collapse else'text_baseline_data/continue_training_save_collapsed.npz'
		if os.path.isfile(save_path)and not get_new_data:load=np.load(save_path);return load['d'],load['e'],load['l']
		out_data,out_exp,out_labels=[],[],[];print('Reading train failed data from {}'.format(path));fnames=[x for x in os.listdir(path)if x.endswith('png')];pat=re.compile('(\\d+)_([^_]+)_([^\\.]*)\\.png');data=[];exp_t=[]
		for fname in fnames:
			mat=re.match(pat,fname);assert mat;data.append(cv2.imread(os.path.join(path,fname),cv2.IMREAD_GRAYSCALE));tmp_char=mat.group(2)
			if len(tmp_char)>1:tmp_char=self.single_char_names[int(tmp_char)]
			assert len(tmp_char)==1 and tmp_char in self.single_char_names;tmp_char_i=self.single_char_names.index(tmp_char);exp_t.append(tmp_char_i)
		exp_t=np.array(exp_t)
		if label_collapse:
			for i in range(len(self.single_char_names)):exp_t[exp_t==i]=self.valid_out.index(self.equal_chars_map[self.single_char_names[i]])
			assert len(np.unique(exp_t))<=77
		else:assert len(np.unique(exp_t))<100
		out_data=np.array(data);out_exp=exp_t;out_labels=[1]*len(data)
		if get_variants:
			variants=self.generate_variations(out_data,variants_num=get_variants)
			for v in variants:out_data=np.vstack((out_data,v));out_exp=np.concatenate((out_exp,exp_t));out_labels+=[1]*len(data)
		if get_false:
			selection=np.unique(exp_t);out_data=np.vstack((out_data,out_data))
			for i in range(len(out_exp)):
				new_exp=random.choice(selection)
				while new_exp==out_exp[i]:new_exp=random.choice(selection)
				out_exp=np.append(out_exp,[new_exp],axis=0)
			out_labels+=[0]*(len(out_data)//2)
		assert len(np.unique(exp_t))<=77
		for v in out_exp:assert 0<=v and v<len(self.single_char_names),v
		for v in out_labels:assert v in[0,1]
		assert len(out_data)==len(out_exp)==len(out_labels);print('Train failed data shape {}'.format(out_data.shape))
		if path=='text_baseline_data/continue_training':np.savez_compressed(save_path,d=out_data,e=out_exp,l=out_labels)
		return out_data,out_exp,out_labels
	def load_az_dataset(self,datasetPath='/home/ss/Documents/VT/html_rewrite/text_baseline_data/a_z_handwritten_data.csv'):
		data=[];labels=[]
		for row in open(datasetPath):row=row.split(',');label=int(row[0]);image=np.array([int(x)for x in row[1:]],dtype='uint8');image=image.reshape((28,28));data.append(image);labels.append(label)
		data=np.array(data,dtype='float32');labels=np.array(labels,dtype='int');return data,labels
	def load_mnist_dataset(self):(trainData,trainLabels),(testData,testLabels)=mnist.load_data();data=np.vstack([trainData,testData]);labels=np.hstack([trainLabels,testLabels]);return data,labels
	def replaceRandom(self,arr,num,val):temp=np.asarray(arr);shape=temp.shape;temp=temp.flatten();inds=np.random.choice(temp.size,size=num);temp[inds]=val;temp=temp.reshape(shape);return temp
	def load_emnist_dataset(self,data_type='by_class'):imgs,labels=extract_training_samples(data_type);imgs=[cv2.resize(image,(32,32))for image in imgs];return imgs,labels
	def processed_single_char_data(self,label_collapse=False,get_new_data=False,get_variants=True):
		save_path='text_baseline_data/full_save.npz'if not label_collapse else'text_baseline_data/full_save_collapsed.npz'
		if os.path.isfile(save_path)and not get_new_data:load=np.load(save_path);return load['d'],load['e'],load['l']
		out_data,out_exp,out_labels=[],[],[];data,exp_t=self.load_single_char_t()
		if label_collapse:
			assert isinstance(exp_t,np.ndarray)
			for i in range(len(self.single_char_names)):exp_t[exp_t==i]=self.valid_out.index(self.equal_chars_map[self.single_char_names[i]])
			assert len(np.unique(exp_t))<90
		out_data=np.array(data);out_exp=np.array(exp_t);out_labels=[1]*len(data)
		if get_variants:
			variants=self.generate_variations(out_data)
			for v in variants:out_data=np.vstack((out_data,v));out_exp=np.concatenate((out_exp,exp_t));out_labels+=[1]*len(data)
		print(data.shape,out_data.shape,out_exp.shape,exp_t.shape,len(out_labels));out_data=np.vstack((out_data,out_data))
		for i in range(len(out_exp)):
			new_exp=random.choice(range(0,len(self.valid_out)))
			while new_exp==out_exp[i]:new_exp=random.choice(range(0,len(self.valid_out)))
			out_exp=np.append(out_exp,[new_exp],axis=0)
		out_labels+=[0]*(len(out_data)//2)
		for v in out_exp:assert-1<v and v<len(self.single_char_names),v
		for v in out_labels:assert v in[0,1]
		assert out_data.shape[0]==out_exp.shape[0]==len(out_labels);np.savez_compressed(save_path,d=out_data,e=out_exp,l=out_labels);return out_data,out_exp,out_labels
	def shape_single_char_img(self,img):assert len(img.shape)==2;larger_val=max(img.shape);smaller_val=min(img.shape);larger_ind=np.argmax(img.shape);smaller_ind=np.argmin(img.shape);to_pad=[[0,0],[0,0]];total_pad=larger_val-smaller_val;before_pad=total_pad//2;to_pad[smaller_ind]=[before_pad,total_pad-before_pad];padded=np.pad(img,to_pad);assert padded.shape[0]==padded.shape[1];resized=cv2.resize(padded,(32,32),interpolation=cv2.INTER_CUBIC);return resized
	def generate_variations(self,data,get_additional=False,variants_num=5):
		positions=[(i,j)for i in range(-variants_num,variants_num)for j in range(-variants_num,variants_num)];positions.remove((0,0));ret=[]
		for(i,j)in positions:tmp=np.roll(data,i,axis=1);tmp=np.roll(tmp,j,axis=2);ret.append(tmp)
		for intensity in range(1,variants_num):tmp=np.where(data==255,255-intensity,data);ret.append(tmp);tmp=np.where(data==0,intensity,data);ret.append(tmp)
		k=variants_num
		if get_additional:k=25
		for repeat in range(k):noise=np.random.normal(0,10,data.shape);tmp=data+noise;ret.append(tmp)
		for repeat in range(k):tmp=self.replaceRandom(data,repeat,0);ret.append(tmp);tmp=self.replaceRandom(data,k*2-repeat,255);ret.append(tmp)
		w,h=data.shape[1:]
		for i in range(3):ret.append(np.array([cv2.resize(img,(w,h))for img in data[:,i:w-i,i:h-i]]));ret.append(np.array([cv2.resize(cv2.copyMakeBorder(img,i,i,i,i,cv2.BORDER_CONSTANT,0),(w,h))for img in data]))
		ret=np.array(ret,dtype='uint8');code.interact(local=dict(locals(),**globals()));return ret
	def load_ocrb_with_other_fonts_false(self,font='ocrb',label_collapse=True,get_new_data=False,get_variants=10):
		tmp_font='ocrb_with_other_fonts';save_path='text_baseline_data/{}/save.npz'.format(tmp_font)if not label_collapse else'text_baseline_data/{}/save_collapsed.npz'.format(tmp_font)
		if os.path.isfile(save_path)and not get_new_data:load=np.load(save_path);return load['d'],load['e'],load['l']
		load_ocrb=False;ocrb_load_path='text_baseline_data/ocrb/save.npz';out_data,out_exp,out_labels=None,None,None
		if os.path.isfile(ocrb_load_path)and load_ocrb:load=np.load(ocrb_load_path);out_data=load['d'];out_exp=load['e'];out_labels=load['l']
		else:
			assert os.path.isdir('text_baseline_data/{}/true'.format(font)),'text_baseline_data/{}/true'.format(font);data,exp_t=self.load_single_char_t(path='text_baseline_data/{}/true'.format(font))
			if label_collapse:
				assert isinstance(exp_t,np.ndarray)
				for i in range(len(self.single_char_names)):exp_t[exp_t==i]=self.valid_out.index(self.equal_chars_map[self.single_char_names[i]])
				assert len(np.unique(exp_t))<90
			out_data=data;out_exp=exp_t;out_labels=[1]*len(data)
			if get_variants:
				variants=self.generate_variations(out_data,variants_num=5)
				for v in variants:out_data=np.vstack((out_data,v));out_exp=np.concatenate((out_exp,exp_t));out_labels+=[1]*len(data)
		num_true_label=len(out_data);false_font=random.choice([x for x in os.listdir('text_baseline_data')if os.path.isdir(os.path.join('text_baseline_data',os.path.join(x,'true')))and x!=font]);f_data,f_exp,_=self.load_single_font(font=false_font,label_collapse=label_collapse,get_new_data=True,get_variants=15,get_false=False);assert len(f_data)>len(out_data);indices=np.random.choice(len(out_data),len(out_data));f_data=f_data[indices];f_exp=f_exp[indices];f_label=[0]*len(f_data);print(out_data.shape,f_data.shape);out_data=np.vstack((out_data,f_data));out_exp=np.concatenate((out_exp,f_exp));out_labels=out_labels+f_label
		for d in out_data:assert d.shape==(32,32)
		for v in out_labels:assert v in[0,1]
		assert out_data.shape[0]==out_exp.shape[0]==len(out_labels),(out_data.shape[0],out_exp.shape[0],len(out_labels));np.savez_compressed(save_path,d=out_data,e=out_exp,l=out_labels);return out_data,out_exp,out_labels
	def load_single_font(self,font='ocrb',label_collapse=False,get_new_data=False,get_variants=True,get_false=True):
		prefix=pathlib.Path(__file__).parent.resolve();save_path='{}/text_baseline_data/{}/save.npz'.format(prefix,font)if not label_collapse else'{}/text_baseline_data/{}/save_collapsed.npz'.format(prefix,font)
		if os.path.isfile(save_path)and not get_new_data:load=np.load(save_path);return load['d'],load['e'],load['l']
		out_data,out_exp,out_labels=[],[],[];assert os.path.isdir('{}/text_baseline_data/{}/true'.format(prefix,font)),'{}/text_baseline_data/{}/true'.format(prefix,font);data,exp_t=self.load_single_char_t(path='{}/text_baseline_data/{}/true'.format(prefix,font))
		if label_collapse:
			assert isinstance(exp_t,np.ndarray)
			for i in range(len(self.single_char_names)):exp_t[exp_t==i]=self.valid_out.index(self.equal_chars_map[self.single_char_names[i]])
			assert len(np.unique(exp_t))<90
		out_data=data;out_exp=exp_t;out_labels=[1]*len(data)
		if get_variants:
			variants=self.generate_variations(out_data,variants_num=get_variants)
			for v in variants:out_data=np.vstack((out_data,v));out_exp=np.concatenate((out_exp,exp_t));out_labels+=[1]*len(data)
		if get_false:
			out_data=np.vstack((out_data,out_data));selection=np.unique(exp_t)
			for i in range(len(out_exp)):
				new_exp=random.choice(selection)
				while new_exp==out_exp[i]:new_exp=random.choice(selection)
				out_exp=np.append(out_exp,[new_exp],axis=0)
			out_labels+=[0]*(len(out_data)//2)
		for d in out_data:assert d.shape==(32,32)
		for v in out_labels:assert v in[0,1]
		assert out_data.shape[0]==out_exp.shape[0]==len(out_labels);np.savez_compressed(save_path,d=out_data,e=out_exp,l=out_labels);return out_data,out_exp,out_labels
	def load_single_char_t(self,path='text_baseline_data/single_char/true_corrected_v2'):
		pattern5=re.compile('([^_]*)_([^_]*)_([^_]*)_([^\\.]*)_([^\\.]*)\\.png');pattern4=re.compile('([^_]*)_([^_]*)_([^_]*)_([^\\.]*)\\.png');pattern3=re.compile('([^_]*)_([^_]*)_([^_]*)\\.png');pattern2=re.compile('');fnames=[x for x in os.listdir(path)if x.endswith('.png')];data=[];labels=[]
		for f in fnames:img=cv2.imread(os.path.join(path,f),cv2.IMREAD_GRAYSCALE);img=cv2.bitwise_not(img);img=self.shape_single_char_img(img);match=re.match(pattern5,f);assert match;target,platform,browser,font_ind,style_i=int(match.group(1)),match.group(2),match.group(3),int(match.group(4)),int(match.group(4));assert platform and browser,f;assert target>-1 and font_ind>-1 and style_i>-1,f;data.append(img);assert target<len(self.single_char_names),target;labels.append(target)
		data=np.array(data,dtype='float32');labels=np.array(labels,dtype='int');return data,labels
	def load_single_char(self):data,labels=self.load_single_char_t();return data,labels
	def load_processed_data(self,percentage=1):
		loaded=np.load('/home/ss/Documents/VT/html_rewrite/text_baseline_data/single_char/complete/save.npz')
		if percentage==1:return loaded['d'],loaded['e'],loaded['l']
		else:return np.append(loaded['d'][:350000],loaded['d'][-350000:],axis=0),np.append(loaded['e'][:350000],loaded['e'][-350000:],axis=0),np.append(loaded['l'][:350000],loaded['l'][-350000:],axis=0)
class CarliniWagnerL2Exception(Exception):0
class CarliniWagnerL2:
	def __init__(self,model_fn,y=None,clip_min=.0,clip_max=1.,targeted=False,binary_search_steps=10,max_iterations=1000,abort_early=True,confidence=.0,initial_const=.01,learning_rate=.005,thr=.5):'\n\t\tThis attack was originally proposed by Carlini and Wagner. It is an\n\t\titerative attack that finds adversarial examples on many defenses that\n\t\tare robust to other attacks.\n\t\tPaper link: https://arxiv.org/abs/1608.04644\n\t\tAt a high level, this attack is an iterative attack using Adam and\n\t\ta specially-chosen loss function to find adversarial examples with\n\t\tlower distortion than other attacks. This comes at the cost of speed,\n\t\tas this attack is often much slower than others.\n\t\t:param model_fn: a callable that takes an input tensor and returns the model logits.\n\t\t:param y: (optional) Tensor with target labels.\n\t\t:param targeted: (optional) Targeted attack?\n\t\t:param clip_min: (optional) float. Minimum float values for adversarial example components.\n\t\t:param clip_max: (optional) float. Maximum float value for adversarial example components.\n\t\t:param binary_search_steps (optional): The number of times we perform binary\n\t\t\t\t\t\t\t\tsearch to find the optimal tradeoff-\n\t\t\t\t\t\t\t\tconstant between norm of the purturbation\n\t\t\t\t\t\t\t\tand confidence of the classification.\n\t\t:param max_iterations (optional): The maximum number of iterations. Setting this\n\t\t\t\t\t\t\t   to a larger value will produce lower distortion\n\t\t\t\t\t\t\t   results. Using only a few iterations requires\n\t\t\t\t\t\t\t   a larger learning rate, and will produce larger\n\t\t\t\t\t\t\t   distortion results.\n\t\t:param abort_early (optional): If true, allows early aborts if gradient descent\n\t\t\t\t\t\tis unable to make progress (i.e., gets stuck in\n\t\t\t\t\t\ta local minimum).\n\t\t:param confidence (optional): Confidence of adversarial examples: higher produces\n\t\t\t\t\t\t   examples with larger l2 distortion, but more\n\t\t\t\t\t\t   strongly classified as adversarial.\n\t\t:param initial_const (optional): The initial tradeoff-constant used to tune the\n\t\t\t\t\t\t  relative importance of the size of the perturbation\n\t\t\t\t\t\t  and confidence of classification.\n\t\t\t\t\t\t  If binary_search_steps is large, the initial\n\t\t\t\t\t\t  constant is not important. A smaller value of\n\t\t\t\t\t\t  this constant gives lower distortion results.\n\t\t:param learning_rate (optional): The learning rate for the attack algorithm.\n\t\t\t\t\t\t  Smaller values produce better results but are\n\t\t\t\t\t\t  slower to converge.\n\t\t';self.model_fn=model_fn;self.prev=None;self.y=y;self.clip_min=clip_min;self.clip_max=clip_max;self.targeted=targeted;self.thr=thr;self.binary_search_steps=binary_search_steps;self.max_iterations=max_iterations;self.abort_early=abort_early;self.learning_rate=learning_rate;self.confidence=confidence;self.initial_const=initial_const;self.optimizer=tf.keras.optimizers.Adam(self.learning_rate);super(CarliniWagnerL2,self).__init__()
	def attack(self,x):'\n\t\tReturns adversarial examples for the tensor.\n\t\t:param x: input tensor.\n\t\t:return: a numpy tensor with the adversarial example.\n\t\t';self.exp_t=x[1];x=x[0];adv_ex=self._attack(x).numpy();return adv_ex
	def _attack(self,x):
		y=self.model_fn([x,self.exp_t]);original_x=tf.cast(x,tf.float32);shape=original_x.shape
		if not y.shape.as_list()[0]==original_x.shape.as_list()[0]:raise CarliniWagnerL2Exception('x and y do not have the same shape!')
		x=original_x;x=x*2.-1.;x=tf.atanh(x*.999999);lower_bound=tf.zeros(shape[:1]);upper_bound=tf.ones(shape[:1])*1e10;const=tf.ones(shape)*self.initial_const
		if const.shape[-1]!=1:const=const[:,:,:,0,np.newaxis]
		assert const.shape[1:]==(32,32,1);best_l2=tf.fill(shape[:1],1e10);best_score=tf.fill(shape[:1],-1);best_score=tf.cast(best_score,tf.int32);best_attack=original_x;modifier=tf.Variable(tf.zeros(shape,dtype=x.dtype),trainable=True)
		for outer_step in range(self.binary_search_steps):
			modifier.assign(tf.zeros(shape,dtype=x.dtype))
			for var in self.optimizer.variables():var.assign(tf.zeros(var.shape,dtype=var.dtype))
			current_best_l2=tf.fill(shape[:1],1e10);current_best_score=tf.fill(shape[:1],-1);current_best_score=tf.cast(current_best_score,tf.int32)
			if self.binary_search_steps>=10 and outer_step==self.binary_search_steps-1:const=upper_bound
			prev=None
			for iteration in range(self.max_iterations):
				x_new,loss,preds,l2_dist=self.attack_step(x,y,modifier,const)
				if self.abort_early and iteration%(self.max_iterations//10 or 1)==0:
					if prev is not None and loss>prev*.9999:break
					prev=loss
				lab=tf.squeeze(y>self.thr);pred_with_conf=preds+self.confidence;pred_with_conf=tf.squeeze(pred_with_conf>self.thr);pred=tf.squeeze(preds>self.thr);pred=tf.cast(pred,tf.int32);mask=tf.math.logical_and(tf.less(l2_dist,current_best_l2),tf.not_equal(pred_with_conf,lab));current_best_l2=set_with_mask(current_best_l2,l2_dist,mask);current_best_score=set_with_mask(current_best_score,pred,mask);mask=tf.math.logical_and(tf.less(l2_dist,best_l2),tf.not_equal(pred_with_conf,lab));best_l2=set_with_mask(best_l2,l2_dist,mask);best_score=set_with_mask(best_score,pred,mask);mask=tf.reshape(mask,[-1,1,1,1]);mask=tf.tile(mask,[1,*best_attack.shape[1:]]);best_attack=set_with_mask(best_attack,x_new,mask)
			lab=tf.squeeze(y>self.thr);lab=tf.cast(lab,tf.int32);upper_mask=tf.math.logical_and(tf.not_equal(best_score,lab),tf.not_equal(best_score,-1));upper_bound=set_with_mask(upper_bound,tf.math.minimum(upper_bound,const),upper_mask);const_mask=tf.math.logical_and(upper_mask,tf.less(upper_bound,1e9));const=set_with_mask(const,(lower_bound+upper_bound)/2.,const_mask);lower_mask=tf.math.logical_not(upper_mask);lower_bound=set_with_mask(lower_bound,tf.math.maximum(lower_bound,const),lower_mask);const_mask=tf.math.logical_and(lower_mask,tf.less(upper_bound,1e9));const=set_with_mask(const,(lower_bound+upper_bound)/2,const_mask);const_mask=tf.math.logical_not(const_mask);const=set_with_mask(const,const*10,const_mask)
		return best_attack
	def attack_step(self,x,y,modifier,const):x_new,grads,loss,preds,l2_dist=self.gradient(x,y,modifier,const);self.optimizer.apply_gradients([(grads,modifier)]);return x_new,loss,preds,l2_dist
	def gradient(self,x,y,modifier,const):
		with tf.GradientTape()as tape:adv_image=modifier+x;x_new=self.clip_tanh(adv_image,clip_min=self.clip_min,clip_max=self.clip_max);preds=self.model_fn([x_new,self.exp_t]);loss,l2_dist=self.loss_fn(x=x,x_new=x_new,y_true=y,y_pred=preds,confidence=self.confidence,const=const,targeted=self.targeted,clip_min=self.clip_min,clip_max=self.clip_max)
		grads=tape.gradient(loss,adv_image);return x_new,grads,loss,preds,l2_dist
	def l2(self,x,y):return tf.reduce_sum(tf.square(x-y),list(range(1,len(x.shape))))
	def loss_fn(self,x,x_new,y_true,y_pred,confidence,const=0,targeted=False,clip_min=0,clip_max=1):
		other=self.clip_tanh(x,clip_min=clip_min,clip_max=clip_max);l2_dist=self.l2(x_new,other);y_true=tf.cast(y_true>self.thr,tf.float32);real=tf.squeeze(y_true*y_pred+(1-y_true)*(1-y_pred),1);other=tf.squeeze(y_true*(1-y_pred)+(1-y_true)*y_pred,1)
		if targeted:loss_1=tf.maximum(.0,other-real+confidence)
		else:loss_1=tf.maximum(.0,real-other+confidence)
		loss_2=tf.reduce_sum(l2_dist);loss_1=tf.reduce_sum(const*loss_1);loss=loss_1+loss_2;return loss,l2_dist
	def clip_tanh(self,x,clip_min,clip_max):return(tf.tanh(x)+1)/2*(clip_max-clip_min)+clip_min
class OurAttacks:
	'\n\tAll attacks here assume x is a length of two and the all attacks are generated for the first item in x\n\t'
	def __init__(self):0
	@tf.function
	def compute_gradient(self,model_fn,loss_fn,x,y,targeted):
		'\n\t\tComputes the gradient of the loss with respect to the input tensor.\n\t\t:param model_fn: a callable that takes an input tensor and returns the model logits.\n\t\t:param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.\n\t\t:param x: input tensor\n\t\t:param y: Tensor with true labels. If targeted is true, then provide the target label.\n\t\t:param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will\n\t\t\t\t\t\ttry to make the label incorrect. Targeted will instead try to move in the\n\t\t\t\t\t\tdirection of being more like y.\n\t\t:return: A tensor containing the gradient of the loss with respect to the input tensor.\n\t\t'
		with tf.GradientTape()as g:
			g.watch(x[0]);loss=loss_fn(labels=y,logits=model_fn(x))
			if targeted:loss=-loss
		grad=g.gradient(loss,x[0]);return grad
	def fast_gradient_method(self,model_fn,x,eps,norm,labels,range_check=True,clip_min=0,clip_max=1):from cleverhans.tf2.utils import optimize_linear;y=model_fn(x);labels=tf.cast(labels,'float32');grad=self.compute_gradient(model_fn,tf.nn.sigmoid_cross_entropy_with_logits,x,y,False);optimal_perturbation=optimize_linear(grad,eps,norm);adv_x=x[0]+optimal_perturbation;adv_x=tf.clip_by_value(adv_x,clip_min,clip_max);return adv_x
	def projected_gradient_descent(self,model_fn,x,eps,eps_iter,nb_iter,norm,labels,loss_fn=tf.nn.sigmoid_cross_entropy_with_logits,clip_min=0,clip_max=1,rand_init=None,rand_minmax=None,sanity_checks=False,targeted=False):
		from cleverhans.tf2.utils import clip_eta,random_lp_vector;assert eps_iter<=eps,(eps_iter,eps)
		if norm==1:raise NotImplementedError("It's not clear that FGM is a good inner loop step for PGD when norm=1, because norm=1 FGM  changes only one pixel at a time. We need  to rigorously test a strong norm=1 PGD before enabling this feature.")
		if norm not in[np.inf,2]:raise ValueError('Norm order must be either np.inf or 2.')
		if loss_fn is None:loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits
		if rand_minmax is None:rand_minmax=eps
		exp=x[1];x=x[0]
		if rand_init:eta=random_lp_vector(tf.shape(x),norm,tf.cast(rand_minmax,x.dtype),dtype=x.dtype)
		else:eta=tf.zeros_like(x)
		eta=clip_eta(eta,norm,eps);adv_x=x+eta
		if clip_min is not None or clip_max is not None:adv_x=tf.clip_by_value(adv_x,clip_min,clip_max)
		i=0
		while i<nb_iter:
			adv_x=self.fast_gradient_method(model_fn,[adv_x,exp],eps_iter,norm,labels,clip_min=clip_min,clip_max=clip_max);eta=adv_x-x;eta=clip_eta(eta,norm,eps);adv_x=x+eta
			if clip_min is not None or clip_max is not None:adv_x=tf.clip_by_value(adv_x,clip_min,clip_max)
			i+=1
		return adv_x
	def cw2(self,model_fn,x,**kwargs):adv_x=CarliniWagnerL2(model_fn,**kwargs).attack(x);return adv_x
	def momentum(self,model_fn,x,eps=.3,eps_iter=.06,nb_iter=10,norm=np.inf,clip_min=None,clip_max=None,y=None,targeted=False,decay_factor=1.):
		from cleverhans.tf2.utils import optimize_linear,clip_eta;y=model_fn(x);momentum=tf.zeros_like(x[0]);adv_x=x[0];i=0
		while i<nb_iter:
			y=np.squeeze(y);y=np.where(y==1,.9999,y);y=y[:,np.newaxis];grad=self.compute_gradient(model_fn,tf.nn.sigmoid_cross_entropy_with_logits,[adv_x,x[1]],y,targeted);red_ind=list(range(1,len(grad.shape)));avoid_zero_div=tf.cast(1e-12,grad.dtype);grad=grad/tf.math.maximum(avoid_zero_div,tf.math.reduce_mean(tf.math.abs(grad),red_ind,keepdims=True));momentum=decay_factor*momentum+grad;optimal_perturbation=optimize_linear(momentum,eps_iter,norm);adv_x=adv_x+optimal_perturbation;adv_x=x[0]+clip_eta(adv_x-x[0],norm,eps)
			if clip_min is not None and clip_max is not None:adv_x=tf.clip_by_value(adv_x,clip_min,clip_max)
			i+=1
		return adv_x
class OurModel:
	def __init__(self):0
	def test_cnn_emnist(self):data_handler=OurData();data,exp,_=data_handler.load_single_font(font='ocrb',get_new_data=True);data=np.array(data,dtype='float32');exp=np.array(exp,dtype='int');data=np.expand_dims(data,axis=-1);data/=255.;le=LabelBinarizer();exp=le.fit_transform(exp);trainX,testX,trainY,testY=train_test_split(data,exp,test_size=.2,stratify=exp,random_state=42);del data,exp;path_model='model_filter.h5';model=CNN_EMNIST().mod1(len(le.classes_));model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam');K.set_value(model.optimizer.lr,.001);h=model.fit(x=trainX,y=trainY,batch_size=128,epochs=10,verbose=1,validation_data=(testX,testY),shuffle=True,callbacks=[ModelCheckpoint(filepath=path_model)])
	def test_resnet(self):
		'\n\t\tTest \n\t\t';print('[INFO] loading datasets...');data_handler=OurData();data,exp,_=data_handler.load_single_font(font='ocrb',get_new_data=True);data=np.array(data,dtype='float32');exp=np.array(exp,dtype='int');data=np.expand_dims(data,axis=-1);data/=255.;le=LabelBinarizer();exp=le.fit_transform(exp);trainX,testX,trainY,testY=train_test_split(data,exp,test_size=.2,stratify=exp,random_state=42);del data,exp;print('[INFO] trainX:{} testX:{} '.format(trainX.shape,testX.shape));print('[INFO] compiling model...');EPOCHS=10;INIT_LR=.001;BS=128;opt=keras.optimizers.Adam(learning_rate=INIT_LR);stages=3,3,3;filters=128,128,256,512;aug=ImageDataGenerator(rotation_range=10,zoom_range=.05,width_shift_range=.1,height_shift_range=.1,shear_range=.15,horizontal_flip=False,fill_mode='nearest');model=ResNet.build(32,32,1,len(le.classes_),stages,filters,reg=.0005);model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy']);model_name='text_baseline_model/test_resnet_v2_{}_{}.h5'.format(stages,filters);print('[INFO] training network...');H=model.fit(aug.flow(trainX,trainY,batch_size=BS),validation_data=(testX,testY),steps_per_epoch=math.ceil(len(trainX)/BS),epochs=EPOCHS,verbose=1);print('[INFO] serializing network... ');model.save(model_name,save_format='h5');labelNames=[l for l in data_handler.single_char_names];print('[INFO] evaluating network...');predictions=model.predict(testX,batch_size=BS);print(classification_report(testY,predictions));images=[]
		for i in np.random.choice(np.arange(0,len(testY)),size=(49,)):
			prediction=model.predict([testX[(np.newaxis,i)],exp[(np.newaxis,i)]])[0];exp_label=labelNames[np.argmax(exp[i])];pred_label=labelNames[prediction];image=(testX[i]*255).astype('uint8');color=0,255,0
			if exp_label!=pred_label:color=0,0,255
			image=cv2.merge([image]*3);image=cv2.resize(image,(96,96),interpolation=cv2.INTER_LINEAR);cv2.putText(image,pred_label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2);images.append(image)
		montage=build_montages(images,(96,96),(7,7))[0];cv2.imwrite('OCR Results.png',montage)
	def load_data(self,dataset,data_handler,label_collapse,get_new_data):
		if dataset.lower()=='full':data,exp,labels=data_handler.processed_single_char_data(label_collapse=label_collapse,get_new_data=get_new_data)
		elif dataset.lower()=='train_failed':data,exp,labels=data_handler.load_train_failed(path='tmp/sample',get_variants=True,label_collapse=label_collapse,get_new_data=get_new_data,get_false=True)
		elif dataset.lower()=='with_other_fonts':data,exp,labels=data_handler.load_ocrb_with_other_fonts_false(font='ocrb',label_collapse=label_collapse,get_new_data=get_new_data,get_variants=10)
		else:data,exp,labels=data_handler.load_single_font(font=dataset,label_collapse=label_collapse,get_new_data=get_new_data)
		data=np.array(data,dtype='float32');exp=np.array(exp,dtype='int');le=None;le=LabelBinarizer()
		if len(np.unique(exp))<77:_,fit_exp,_=data_handler.load_single_font(font='ocrb',get_variants=False,label_collapse=label_collapse,get_new_data=get_new_data);le.fit(fit_exp);exp=le.transform(exp)
		else:exp=le.fit_transform(exp)
		data=np.expand_dims(data,axis=-1);data/=255.;labels=np.array(labels,dtype='int');return data,exp,labels,le
	def train_with_other_fonts(self,model_complexity='mod2',label_collapse=True,continue_training=False,custermized_thr=.99):
		print('[INFO] loading datasets...');data_handler=OurData();data,exp,labels,le=self.load_data('with_other_fonts',data_handler,label_collapse=label_collapse,get_new_data=True);trainX,testX,train_exp,test_exp,trainY,testY=train_test_split(data,exp,labels,test_size=.2,stratify=labels,random_state=42);del data,exp,labels;print('[INFO] compiling model...');EPOCHS=20;INIT_LR=.1;BS=128;opt=keras.optimizers.Adam();model_name='text_baseline_model/ocrb_mod2_True_0.99_with_other_fonts.h5';model=None
		if continue_training and os.path.isfile(model_name):model=keras.models.load_model(model_name)
		if model is None:
			if model_complexity in['s','m','c']:
				if model_complexity in's':stages=3,3;filters=32,32,64
				elif model_complexity=='m':stages=3,3,3;filters=64,64,128,256
				else:stages=3,3,3;filters=128,128,256,512
				model=RV.build(32,32,1,len(le.classes_)if le is not None else 1,stages,filters,reg=.0005);model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod1':model=CNN_EMNIST().mod1_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod2':model=CNN_EMNIST().mod2_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod3':model=CNN_EMNIST().mod3_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod4':model=CNN_EMNIST().mod4_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod5':model=CNN_EMNIST().mod5_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod6':model=CNN_EMNIST().mod6_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod7':model=CNN_EMNIST().mod7_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
		print('[INFO] training network...');H=model.fit(x=[trainX,train_exp],y=trainY,validation_data=([testX,test_exp],testY),steps_per_epoch=math.ceil(len(trainX)/BS),epochs=EPOCHS,shuffle=True,verbose=1);print('[INFO] serializing network... ');model.save(model_name,save_format='h5');self.test(testX,test_exp,testY,model,labelNames=data_handler.single_char_names if not label_collapse else data_handler.valid_out,BS=BS)
	def train(self,model_complexity='c',dataset='ocrb',label_collapse=False,get_new_data=False,continue_training=False,custermized_thr=.5):
		print('[INFO] loading datasets...');data_handler=OurData();data,exp,labels,le=self.load_data(dataset,data_handler,label_collapse,get_new_data);trainX,testX,train_exp,test_exp,trainY,testY=train_test_split(data,exp,labels,test_size=.2,stratify=labels,random_state=42);del data,exp,labels
		if gtpu:strategy=distribute_utils.get_distribution_strategy(distribution_strategy='tpu',num_gpus=True,tpu_address='test');strategy_scope=distribute_utils.get_strategy_scope(strategy)
		print('[INFO] compiling model...');EPOCHS=20;INIT_LR=.1;BS=128;opt=keras.optimizers.Adam()
		if dataset=='train_failed':model_name='text_baseline_model/ocrb_mod2_True_train_failed.h5';continue_training=True
		elif custermized_thr==.5:model_name='text_baseline_model/{}_{}_{}.h5'.format(dataset,model_complexity,label_collapse)
		else:model_name='text_baseline_model/{}_{}_{}_{}.h5'.format(dataset,model_complexity,label_collapse,custermized_thr)
		model=None
		if continue_training and os.path.isfile(model_name):model=keras.models.load_model(model_name)
		if model is None:
			if model_complexity in['s','m','c']:
				if model_complexity in's':stages=3,3;filters=32,32,64
				elif model_complexity=='m':stages=3,3,3;filters=64,64,128,256
				else:stages=3,3,3;filters=128,128,256,512
				model=RV.build(32,32,1,len(le.classes_)if le is not None else 1,stages,filters,reg=.0005);model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod1':model=CNN_EMNIST().mod1_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod2':model=CNN_EMNIST().mod2_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod3':model=CNN_EMNIST().mod3_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod4':model=CNN_EMNIST().mod4_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod5':model=CNN_EMNIST().mod5_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod6':model=CNN_EMNIST().mod6_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
			elif model_complexity=='mod7':model=CNN_EMNIST().mod7_tf2(len(le.classes_));model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(threshold=custermized_thr)])
		print('[INFO] training network...');H=model.fit(x=[trainX,train_exp],y=trainY,validation_data=([testX,test_exp],testY),steps_per_epoch=math.ceil(len(trainX)/BS),epochs=EPOCHS,shuffle=True,verbose=1);print('[INFO] serializing network... ');model.save(model_name,save_format='h5');self.test(testX,test_exp,testY,model,labelNames=data_handler.single_char_names if not label_collapse else data_handler.valid_out,BS=BS)
	def test(self,testX,exp,testY,model,labelNames='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',BS=128,custermized_thr=.5):
		if type(labelNames)==str:labelNames=[l for l in labelNames]
		print('[INFO] evaluating network...');predictions=model.predict([testX,exp],batch_size=BS);print(testY.shape,predictions.shape);predictions=(predictions>custermized_thr).astype(np.uint8).flatten();print(classification_report(testY,predictions,digits=4));images=[]
		for i in np.random.choice(np.arange(0,len(testY)),size=(49,)):
			prediction=model.predict([testX[(np.newaxis,i)],exp[(np.newaxis,i)]]);prediction=1 if prediction[0]>custermized_thr else 0;label=labelNames[np.argmax(exp[i])];image=(testX[i]*255).astype('uint8');color=0,255,0
			if prediction!=testY[i]:color=0,0,255
			image=cv2.merge([image]*3);image=cv2.resize(image,(96,96),interpolation=cv2.INTER_LINEAR);cv2.putText(image,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2);images.append(image)
		montage=build_montages(images,(96,96),(7,7))[0];cv2.imwrite('OCR Results.png',montage)
	def load_and_test(self,path,font='ocrb',custermized_thr=.5):
		handler=OurData();model=keras.models.load_model(path);label_collapse='True'in path
		if'train_failed'in path:data,exp,labels=handler.load_train_failed(path='tmp/sample',label_collapse=label_collapse,get_new_data=True,get_variants=False,get_false=False)
		elif font=='full':data,exp,labels=handler.processed_single_char_data(label_collapse=label_collapse,get_new_data=True);data,_,exp,full_exp,labels,_=train_test_split(data,exp,labels,test_size=.8,stratify=labels,random_state=42)
		else:data,exp,labels=handler.load_single_font(font=font,label_collapse=label_collapse,get_new_data=True)
		data=np.array(data,dtype='float32');exp=np.array(exp,dtype='int');le=None;le=LabelBinarizer()
		if len(np.unique(exp)<77):_,fit_exp,_=handler.load_single_font(font='ocrb',label_collapse=True,get_new_data=False);le.fit(fit_exp);exp=le.transform(exp)
		else:exp=le.fit_transform(exp)
		data/=255.;labels=np.array(labels,dtype='int');print('Testing dataset has shape: {}'.format(data.shape));self.test(data,exp,labels,model,labelNames=handler.single_char_names if not label_collapse else handler.valid_out,BS=128,custermized_thr=custermized_thr)
	def load_saved_model(self,h5_path):
		fname_no_ext,ext=os.path.splitext(os.path.basename(h5_path));json_config_name=os.path.join(os.path.dirname(h5_path),fname_no_ext+'.json')
		if os.path.isfile(json_config_name):
			with open(json_config_name,'r')as inf:loaded_model_json=inf.read()
			model=keras.models.model_from_json(loaded_model_json);model.load_weights(h5_path)
		else:model=keras.models.load_model(h5_path)
		return model
	def adjust(self,adv_x,EPS,NORM,data):
		adv_x=np.array(adv_x,copy=True);channel=1 if len(adv_x.shape)<4 else adv_x.shape[3];adv_x=adv_x*255;adv_x=np.rint(adv_x);adv_x[adv_x>255]=255;adv_x[adv_x<0]=0;adv_x=tf.cast(adv_x,tf.uint8);adv_x=tf.cast(adv_x,tf.float32);adv_x=adv_x/255;assert np.sum(adv_x>1)==0;assert np.sum(adv_x<0)==0
		if NORM==2:return adv_x
		elif NORM==np.inf:return adv_x
		assert False
	def resnet_robustness_evaluation(self,model_num=1,EPS=.3,NORM=2):
		gpus=tf.config.list_physical_devices('GPU')
		for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
		model_path='text_baseline_model/resnet/handwriting.model';model=keras.models.load_model(model_path);handler=OurData();data,labels=handler.load_mnist_dataset();data,_,labels,_=train_test_split(data,labels,test_size=.998,stratify=labels,random_state=42);data=[cv2.resize(image,(32,32))for image in data];data=np.array(data,dtype='float32');labels=labels.astype('int32');num_samples=120;data=data[:num_samples];labels=labels[:num_samples];data=np.expand_dims(data,axis=-1);data/=255.;test_acc_clean=tf.metrics.SparseCategoricalAccuracy();test_acc_fgsm=tf.metrics.SparseCategoricalAccuracy();test_acc_pgd=tf.metrics.SparseCategoricalAccuracy();test_acc_mom=tf.metrics.SparseCategoricalAccuracy();test_acc_cw2=tf.metrics.SparseCategoricalAccuracy();from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent;from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method;from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method;from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2;print('\n{} {} {}'.format(model_path,EPS,NORM));from autoattack import utils_tf2;from autoattack import AutoAttack;import torch;data=torch.from_numpy(np.transpose(data,(0,3,1,2))).float().cuda();labels=torch.from_numpy(labels).float();thresh=.5;model_adapted=utils_tf2.ModelAdapter(model,num_classes=36);adversary=AutoAttack(model_adapted,norm='L{}'.format(NORM),eps=EPS,version='standard',is_tf_model=True,exp=None,verbose=1,attacks_to_run=['fab','apgd-ce'],thresh=.5);assert len(data)==num_samples;assert len(labels)==num_samples;x_adv,ress=adversary.run_standard_evaluation_individual(data,labels,bs=10);attack_results={}
		for(attack_name,advs)in x_adv.items():outputs=ress[attack_name][0].cpu().numpy()[:,1];np_x_adv=np.moveaxis(advs.cpu().numpy(),1,3);y_pred=model(np_x_adv);test_acc_unadjusted=tf.metrics.SparseCategoricalAccuracy();test_acc_unadjusted(labels,y_pred);adjusted=self.adjust(np_x_adv,EPS,NORM,data);test_acc_adjusted=tf.metrics.SparseCategoricalAccuracy();y_pred2=model(adjusted);test_acc_adjusted(labels,y_pred2);attack_results[attack_name]=test_acc_unadjusted.result()*100,test_acc_adjusted.result()*100
		for(attack_name,accu)in attack_results.items():print('{} adjusted: {:.2f} {:.2f}'.format(attack_name,accu[0],accu[1]))
	def check_gradient(self,x_fgm,labels,model_fn,is_cw2=False):
		'\n\t\tCheck if gradient of the inputs are zero\n\t\t';input=x_fgm;exp_t=labels;y=model_fn([input,exp_t])
		if not is_cw2:grad=OurAttacks().compute_gradient(model_fn,tf.nn.sigmoid_cross_entropy_with_logits,[input,exp_t],y,False)
		else:grad=OurAttacks().compute_gradient(model_fn,tf.nn.sigmoid_cross_entropy_with_logits,[input,exp_t],y,False)
		return np.sum(grad,axis=(1,2,3))!=0
	def robustness_evaluation(self,path,write_output=False,dataset='full',EPS=.3,NORM=np.inf,custermized_thr=.5):
		model=keras.models.load_model(path);test_acc_clean=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_fgsm=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_pgd=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_cw2=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_mom=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_fgsm2=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_pgd2=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_cw22=tf.metrics.BinaryAccuracy(threshold=custermized_thr);test_acc_mom2=tf.metrics.BinaryAccuracy(threshold=custermized_thr);run_cw2=EPS==.5019 and NORM==np.inf or EPS==3 and NORM==2;handler=OurData();label_collapse=True if'True'in path else False;attack_handler=OurAttacks();full_exp=None
		if dataset=='test':data,exp,labels=handler.load_test_dataset(label_collapse=label_collapse)
		elif dataset=='full':data,exp,labels=handler.processed_single_char_data(label_collapse=label_collapse,get_variants=False,get_new_data=True);data,_,exp,full_exp,labels,_=train_test_split(data,exp,labels,test_size=.8,stratify=labels,random_state=42)
		elif dataset.lower()=='with_other_fonts':data,exp,labels=handler.load_ocrb_with_other_fonts_false(font='ocrb',label_collapse=label_collapse,get_new_data=False,get_variants=10);full_exp=exp
		else:data,exp,labels=handler.load_single_font(font=dataset,label_collapse=label_collapse,get_variants=False,get_new_data=True);full_exp=exp
		labels=np.array(labels);ind=labels==0;data=data[ind][:120];exp=exp[ind][:120];target_char_names=handler.valid_out if label_collapse else list(handler.single_char_names);char_exp_names=np.array(target_char_names)[exp];labels=labels[ind][:120];assert np.sum(labels==1)==0;data=np.array(data,dtype='float32');exp=np.array(exp,dtype='int');le=None;le=LabelBinarizer();le.fit(full_exp if full_exp is not None else exp);exp=le.transform(exp);data=np.expand_dims(data,axis=-1);data/=255.;labels=np.array(labels,dtype='int');labels=np.expand_dims(labels,axis=-1);y_pred=model([data,exp]);test_acc_clean(labels,y_pred);begin=time.time();x_fgm=attack_handler.fast_gradient_method(model,[data,exp],EPS,NORM,labels,range_check=False);time_fgm=time.time()-begin;count_fgm_before=np.count_nonzero(np.sum(x_fgm-data,axis=(1,2,3))==0);x_fgm2=self.adjust(x_fgm,EPS,NORM,data);count_fgm_after=np.count_nonzero(np.sum(x_fgm2-data,axis=(1,2,3))==0);y_pred_fgm=model([x_fgm,exp]);y_pred_fgm2=model([x_fgm2,exp]);test_acc_fgsm(labels,y_pred_fgm);test_acc_fgsm2(labels,y_pred_fgm2);begin=time.time();x_pgd=attack_handler.projected_gradient_descent(model,[data,exp],EPS,EPS*.1,10,NORM,labels);time_pgd=time.time()-begin;count_pgd_before=np.count_nonzero(np.sum(x_pgd-data,axis=(1,2,3))==0);x_pgd2=self.adjust(x_pgd,EPS,NORM,data);count_pgd_after=np.count_nonzero(np.sum(x_pgd2-data,axis=(1,2,3))==0);y_pred_pgd=model([x_pgd,exp]);y_pred_pgd2=model([x_pgd2,exp]);test_acc_pgd(labels,y_pred_pgd);test_acc_pgd2(labels,y_pred_pgd);begin=time.time();x_mom=attack_handler.momentum(model,[data,exp],eps=EPS,eps_iter=EPS*.5,nb_iter=10,norm=NORM);time_mom=time.time()-begin;count_mom_before=np.count_nonzero(np.sum(x_mom-data,axis=(1,2,3))==0);x_mom2=self.adjust(x_mom,EPS,NORM,data);count_mom_after=np.count_nonzero(np.sum(x_mom2-data,axis=(1,2,3))==0);y_pred_mom=model([x_mom,exp]);y_pred_mom2=model([x_mom2,exp]);test_acc_mom(labels,y_pred_mom);test_acc_mom2(labels,y_pred_mom);x_cw2,x_cw22,y_pred_cw2,y_pred_cw22=[],[],[],[];count_cw2=0
		if run_cw2:begin=time.time();x_cw2=attack_handler.cw2(model,[data,exp],thr=custermized_thr);time_cw2=time.time()-begin;count_cw2_before=np.count_nonzero(np.sum(x_cw2-data,axis=(1,2,3))==0);x_cw22=self.adjust(x_cw2,EPS,NORM,data);count_cw2_after=np.count_nonzero(np.sum(x_cw22-data,axis=(1,2,3))==0);y_pred_cw2=model([x_cw2,exp]);y_pred_cw22=model([x_cw22,exp]);test_acc_cw2(labels,y_pred_cw2);test_acc_cw22(labels,y_pred_cw22)
		if run_cw2:print('\n{}_{}_{}_{}: FGM:{}/{} PGD:{}/{} MOM:{}/{} CW2:{}/{} out of {}'.format(os.path.basename(path),dataset,EPS,NORM,count_fgm_before,count_fgm_after,count_pgd_before,count_pgd_after,count_mom_before,count_mom_after,count_cw2_before,count_cw2_after,len(y_pred)))
		else:print('\n{}_{}_{}_{}: FGM:{}/{} PGD:{}/{} MOM:{}/{} out of {}'.format(os.path.basename(path),dataset,EPS,NORM,count_fgm_before,count_fgm_after,count_pgd_before,count_pgd_after,count_mom_before,count_mom_after,len(y_pred)))
		print('Clean: {:.2f}'.format(test_acc_clean.result()*100));print('FGM: {:.2f} {:.2f}, {:.2f}s'.format(test_acc_fgsm.result()*100,test_acc_fgsm2.result()*100,time_fgm));print('PGD: {:.2f} {:.2f}, {:.2f}s'.format(test_acc_pgd.result()*100,test_acc_pgd2.result()*100,time_pgd));print('MOM: {:.2f} {:.2f}, {:.2f}s'.format(test_acc_mom.result()*100,test_acc_mom2.result()*100,time_mom))
		if run_cw2:print('CW2: {:.2f} {:.2f}, {:.2f}s'.format(test_acc_cw2.result()*100,test_acc_cw22.result()*100,time_cw2))
		run_validation=False
		if run_validation:
			for i in range(len(x_fgm)):diff=np.linalg.norm((x_fgm[i]-data[i]).numpy().reshape(-1));assert diff<EPS
			for i in range(len(x_pgd)):
				diff=np.linalg.norm((x_pgd[i]-data[i]).numpy().reshape(-1))
				if diff>EPS:print(diff)
			for i in range(len(x_mom)):
				diff=np.linalg.norm((x_mom[i]-data[i]).numpy().reshape(-1))
				if diff>EPS:print(diff)
			for i in range(len(x_cw2)):
				diff=np.linalg.norm((x_cw2[i]-data[i]).numpy().reshape(-1))
				if diff>EPS:print(diff)
		if write_output:
			adv_out_path='text_baseline_adv/{}_{}_{}_{}'.format(os.path.basename(path),dataset,EPS,NORM)
			if not os.path.isdir(adv_out_path):os.mkdir(adv_out_path)
			to_write_data=[data,x_fgm,x_pgd,x_mom,x_cw2];keeps=[np.sum(x_fgm-data,axis=(1,2,3))!=0,np.sum(x_pgd-data,axis=(1,2,3))!=0,np.sum(x_mom-data,axis=(1,2,3))!=0,np.sum(x_cw2-data,axis=(1,2,3))!=0];names=['orig','fgm','pgd','mom','cw2'];ress=[y_pred_fgm,y_pred_pgd,y_pred_mom,y_pred_cw2];exp=[]
			for i in range(len(to_write_data)):
				if i>0:keep=keeps[i-1];res=ress[i-1]
				out_data=to_write_data[i];out_dir=os.path.join(adv_out_path,names[i])
				if os.path.isdir(out_dir):shutil.rmtree(out_dir)
				os.mkdir(out_dir)
				for j in range(len(out_data)):
					tmp=out_data[j]
					if type(tmp)!=np.ndarray:tmp=tmp.numpy()
					tmp=tmp*255;tmp=tmp.astype(np.uint8)
					if i==0:cv2.imwrite('{}/{}_{}.png'.format(out_dir,j,char_exp_names[j]),cv2.bitwise_not(tmp))
					else:cv2.imwrite('{}/{}_{}_{}_{:.2f}.png'.format(out_dir,j,char_exp_names[j],keep[j],res[j][0]*100),cv2.bitwise_not(tmp))
	def auto_attack(self,model_path,dataset='roboto-slab',label_collapse=True,EPS=1,NORM=2,attacks_to_run=['fab','apgd-ce'],write_output=True,thresh=.5):
		from autoattack import utils_tf2,AutoAttack;handler=OurData();data,exp,labels,le=self.load_data(dataset,handler,label_collapse,get_new_data=False);labels=np.array(labels);ind=labels==0;num_samples=120;data=np.array(data[ind][:num_samples]);orig=np.array(data,copy=True);exp=np.array(exp[ind][:num_samples]);labels=np.array(labels[ind][:num_samples]);target_char_names=handler.valid_out if label_collapse else list(handler.single_char_names);char_exp_names=np.array(target_char_names)[np.argmax(exp,axis=1).astype('uint8')];import torch;data=torch.from_numpy(np.transpose(data,(0,3,1,2))).float().cuda();labels=torch.from_numpy(labels).float();gpus=tf.config.list_physical_devices('GPU')
		for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
		model=keras.models.load_model(model_path);print('\n{} {} {} {}'.format(model_path,EPS,NORM,attacks_to_run));model_adapted=utils_tf2.ModelAdapter(model,num_classes=2);adversary=AutoAttack(model_adapted,norm='L{}'.format(NORM),eps=EPS,version='standard',is_tf_model=True,exp=exp,verbose=1,attacks_to_run=attacks_to_run,thresh=thresh);x_adv,ress=adversary.run_standard_evaluation_individual(data,labels,bs=num_samples);attack_results={}
		for(attack_name,advs)in x_adv.items():
			outputs=ress[attack_name][0].cpu().numpy()[:,1];np_x_adv=np.moveaxis(advs.cpu().numpy(),1,3);y_pred=model([np_x_adv,exp]);test_acc_unadjusted=tf.metrics.BinaryAccuracy(threshold=thresh);test_acc_unadjusted(labels,y_pred);adjusted=self.adjust(np_x_adv,EPS,NORM,data);test_acc_adjusted=tf.metrics.BinaryAccuracy(threshold=thresh);y_pred2=model([adjusted,exp]);test_acc_adjusted(labels,y_pred2);attack_results[attack_name]=test_acc_unadjusted.result()*100,test_acc_adjusted.result()*100
			if write_output:
				adv_out_path='text_baseline_adv/{}_{}_{}_{}/{}'.format(os.path.basename(model_path),dataset,EPS,NORM,attack_name)
				if os.path.isdir(adv_out_path):shutil.rmtree(adv_out_path)
				os.makedirs(adv_out_path)
				for j in range(len(np_x_adv)):
					tmp=np_x_adv[j]
					if type(tmp)!=np.ndarray:tmp=tmp.numpy()
					tmp=(tmp*255).astype(np.uint8);cv2.imwrite('{}/{}_{}_{:.2f}.png'.format(adv_out_path,j,char_exp_names[j],outputs[j]*100),cv2.bitwise_not(tmp))
				adv_out_path='text_baseline_adv/{}_{}_{}_{}/{}'.format(os.path.basename(model_path),dataset,EPS,NORM,'orig_fab')
				if os.path.isdir(adv_out_path):shutil.rmtree(adv_out_path)
				os.makedirs(adv_out_path)
				for j in range(len(orig)):
					tmp=orig[j]
					if type(tmp)!=np.ndarray:tmp=tmp.numpy()
					tmp=(tmp*255).astype(np.uint8);cv2.imwrite('{}/{}_{}.png'.format(adv_out_path,j,char_exp_names[j]),cv2.bitwise_not(tmp))
		for(attack_name,res)in attack_results.items():print('{} adjusted: {:.2f} {:.2f}'.format(attack_name,res[0],res[1]))
	def auto_attack(self,model_path,dataset='roboto-slab',label_collapse=True,EPS=1,NORM=2,attacks_to_run=['fab','apgd-ce'],write_output=True,thresh=.5):
		from autoattack import utils_tf2,AutoAttack;handler=OurData();data,exp,labels,le=self.load_data(dataset,handler,label_collapse,get_new_data=False);labels=np.array(labels);ind=labels==0;num_samples=120;data=np.array(data[ind][:num_samples]);orig=np.array(data,copy=True);exp=np.array(exp[ind][:num_samples]);labels=np.array(labels[ind][:num_samples]);target_char_names=handler.valid_out if label_collapse else list(handler.single_char_names);char_exp_names=np.array(target_char_names)[np.argmax(exp,axis=1).astype('uint8')];import torch;data=torch.from_numpy(np.transpose(data,(0,3,1,2))).float().cuda();labels=torch.from_numpy(labels).float();gpus=tf.config.list_physical_devices('GPU')
		for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
		model=keras.models.load_model(model_path);print('\n{} {} {} {}'.format(model_path,EPS,NORM,attacks_to_run));model_adapted=utils_tf2.ModelAdapter(model,num_classes=2);adversary=AutoAttack(model_adapted,norm='L{}'.format(NORM),eps=EPS,version='standard',is_tf_model=True,exp=exp,verbose=1,attacks_to_run=attacks_to_run,thresh=thresh);x_adv,ress=adversary.run_standard_evaluation_individual(data,labels,bs=num_samples);attack_results={}
		for(attack_name,advs)in x_adv.items():
			outputs=ress[attack_name][0].cpu().numpy()[:,1];np_x_adv=np.moveaxis(advs.cpu().numpy(),1,3);y_pred=model([np_x_adv,exp]);test_acc_unadjusted=tf.metrics.BinaryAccuracy(threshold=thresh);test_acc_unadjusted(labels,y_pred);adjusted=self.adjust(np_x_adv,EPS,NORM,data);test_acc_adjusted=tf.metrics.BinaryAccuracy(threshold=thresh);y_pred2=model([adjusted,exp]);test_acc_adjusted(labels,y_pred2);attack_results[attack_name]=test_acc_unadjusted.result()*100,test_acc_adjusted.result()*100
			if write_output:
				adv_out_path='text_baseline_adv/{}_{}_{}_{}/{}'.format(os.path.basename(model_path),dataset,EPS,NORM,attack_name)
				if os.path.isdir(adv_out_path):shutil.rmtree(adv_out_path)
				os.makedirs(adv_out_path)
				for j in range(len(np_x_adv)):
					tmp=np_x_adv[j]
					if type(tmp)!=np.ndarray:tmp=tmp.numpy()
					tmp=(tmp*255).astype(np.uint8);cv2.imwrite('{}/{}_{}_{:.2f}.png'.format(adv_out_path,j,char_exp_names[j],outputs[j]*100),cv2.bitwise_not(tmp))
				adv_out_path='text_baseline_adv/{}_{}_{}_{}/{}'.format(os.path.basename(model_path),dataset,EPS,NORM,'orig_fab')
				if os.path.isdir(adv_out_path):shutil.rmtree(adv_out_path)
				os.makedirs(adv_out_path)
				for j in range(len(orig)):
					tmp=orig[j]
					if type(tmp)!=np.ndarray:tmp=tmp.numpy()
					tmp=(tmp*255).astype(np.uint8);cv2.imwrite('{}/{}_{}.png'.format(adv_out_path,j,char_exp_names[j]),cv2.bitwise_not(tmp))
		for(attack_name,res)in attack_results.items():print('{} adjusted: {:.2f} {:.2f}'.format(attack_name,res[0],res[1]))
	def cw_robustness(self,model_path,dataset='ocrb'):
		from nn_robust_attacks.l0_attack import CarliniL0;from nn_robust_attacks.l2_attack import CarliniL2;from nn_robust_attacks.li_attack import CarliniLi
		with tf.Session()as sess:
			handler=OurData();data,full_exp,labels=handler.load_single_font(font=dataset,label_collapse=label_collapse,get_variants=True,get_new_data=False);labels=np.array(labels);ind=labels==0;num_samples=120;data=np.array(data[ind][:num_samples]);exp=np.array(full_exp[ind][:num_samples]);labels=np.array(labels[ind][:num_samples]);data=np.array(data,dtype='float32');exp=np.array(exp,dtype='int');le=None;le=LabelBinarizer();le.fit(full_exp if full_exp is not None else exp);exp=le.transform(exp);data=np.expand_dims(data,axis=-1);data/=255.;labels=np.array(labels,dtype='int');model=keras.models.load_model(model_path);attack=CarliniL2(sess,model,batch_size=9,max_iterations=1000,confidence=0);timestart=time.time();adv=attack.attack([data,exp],labels);timeend=time.time();print('Took',timeend-timestart,'seconds to run',len(data),'samples.')
			for i in range(len(adv)):print('Classification:',model.model.predict(adv[i:i+1]));print('Total distortion:',np.sum((adv[i]-inputs[i])**2)**.5)
def trace(frame,event,arg):
	if os.path.basename(frame.f_code.co_filename)=='retrain.py':print('%s, %s:%d'%(event,frame.f_code.co_filename,frame.f_lineno))
	return trace
if __name__=='__main__':
	if gtpu:flags_core.define_base(clean=True,num_gpu=True,train_epochs=True,epochs_between_evals=True,distribution_strategy=True);flags_core.define_device();flags_core.define_distribution()
	ourModel=OurModel();ds='full';ourModel.robustness_evaluation('text_baseline_model/{}_mod2_True.h5'.format(ds),write_output=True,dataset=ds,EPS=3,NORM=2)