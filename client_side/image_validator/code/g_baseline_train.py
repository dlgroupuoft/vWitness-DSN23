import random,cv2,code,numpy as np,os,tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input,Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import math
from imutils import build_montages
import progressbar,retrain as t_baseline,shutil
class MyData:
	def __init__(self):self.ciphar_path='g_baseline_data/cifar_10';self.material_path='g_baseline_data/material36';self.t_model=t_baseline.OurData();self.valid_out=self.t_model.valid_out
	def load_ciphar(self,num=10,load=True):
		save_path='g_baseline_data/save/cifar_{}.npz'.format(num-1);assert num<=10 and num>=1
		if load and os.path.isfile(save_path):
			imgs1=None;imgs2=None;l=None
			for i in range(num):
				loaded=np.load('g_baseline_data/save/cifar_{}.npz'.format(i),allow_pickle=True)
				if imgs1 is None:imgs1,imgs2,l=loaded['imgs1'],loaded['imgs2'],loaded['l']
				else:imgs1=np.vstack((imgs1,loaded['imgs1']));imgs2=np.vstack((imgs2,loaded['imgs2']));l=np.concatenate((l,loaded['l']))
			return imgs2,imgs1,l
		(_,_),(x_test,_)=cifar10.load_data();bar=progressbar.ProgressBar();imgs1=None;imgs2=None;labels=[]
		for num_i in bar(range(num)):
			dataset=x_test[num_i*1000:(num_i+1)*1000];assert len(dataset)==1000;variants=self.generate_variants(dataset);randomized=np.array(dataset,copy=True);np.random.shuffle(randomized);assert np.sum(np.sum(randomized-dataset,axis=(0,1,2))==0)==0
			if imgs1 is None:imgs1=dataset
			else:imgs1=np.vstack((imgs1,dataset))
			if imgs2 is None:imgs2=dataset
			else:imgs2=np.vstack((imgs2,dataset))
			labels+=[1]*len(dataset)
			for k in range(len(variants)):one_var=variants[k];assert variants[k].shape==dataset.shape;imgs1=np.vstack((imgs1,one_var));imgs2=np.vstack((imgs2,dataset));labels+=[1]*len(dataset);imgs1=np.vstack((imgs1,one_var));imgs2=np.vstack((imgs2,randomized));labels+=[0]*len(one_var)
			save_path='g_baseline_data/save/cifar_{}.npz'.format(num_i);np.savez_compressed(save_path,imgs1=imgs1,imgs2=imgs2,l=labels)
		return imgs1,imgs2,labels
	def read_icon(self,icon_path):icon=cv2.imread(icon_path,cv2.IMREAD_UNCHANGED);trans_mask=icon[:,:,3]==0;icon[trans_mask]=[255,255,255,255];icon=cv2.cvtColor(icon,cv2.COLOR_BGRA2BGR);return icon
	def load_cifar_with_text(self,num=0,load=False):
		save_path='g_baseline_data/save/cifar_with_text_{}.npz'.format(num-1);assert num<=10 and num>=1
		if load and os.path.isfile(save_path):
			imgs1=None;imgs2=None;l=None
			for i in range(num):
				loaded=np.load('g_baseline_data/save/cifar_{}.npz'.format(i),allow_pickle=True)
				if imgs1 is None:imgs1,imgs2,l=loaded['imgs1'],loaded['imgs2'],loaded['l']
				else:imgs1=np.vstack((imgs1,loaded['imgs1']));imgs2=np.vstack((imgs2,loaded['imgs2']));l=np.concatenate((l,loaded['l']))
			return imgs2,imgs1,l
		(_,_),(x_test,_)=cifar10.load_data();bar=progressbar.ProgressBar();imgs1=None;imgs2=None;labels=[]
		for num_i in bar(range(num)):
			dataset=x_test[num_i*1000:(num_i+1)*1000];assert len(dataset)==1000;variants=self.generate_variants(dataset);randomized=np.array(dataset,copy=True);np.random.shuffle(randomized);assert np.sum(np.sum(randomized-dataset,axis=(0,1,2))==0)==0;with_text=np.array(dataset,copy=True);color=0,0,0
			for i in range(len(with_text)):with_text[i]=cv2.putText(with_text[i],random.choice(self.valid_out),(0,10),cv2.FONT_HERSHEY_SIMPLEX,.25,color,1)
			if imgs1 is None:imgs1=dataset
			else:imgs1=np.vstack((imgs1,dataset))
			if imgs2 is None:imgs2=dataset
			else:imgs2=np.vstack((imgs2,dataset))
			labels+=[1]*len(dataset)
			for k in range(len(variants)):one_var=variants[k];assert variants[k].shape==dataset.shape;imgs1=np.vstack((imgs1,one_var));imgs2=np.vstack((imgs2,dataset));labels+=[1]*len(dataset);imgs1=np.vstack((imgs1,one_var));cv2.putText(one_var);imgs2=np.vstack((imgs2,randomized));labels+=[0]*len(one_var)
			save_path='g_baseline_data/save/cifar_{}.npz'.format(num_i);np.savez_compressed(save_path,imgs1=imgs1,imgs2=imgs2,l=labels)
		return imgs1,imgs2,labels
	def load_material(self,load=True):
		save_path='g_baseline_data/save/material.npz'
		if load and os.path.isfile(save_path):loaded=np.load(save_path,allow_pickle=True);return loaded['imgs2'],loaded['imgs1'],loaded['l']
		dataset=[]
		for fname in os.listdir(self.material_path):dataset.append(self.read_icon(os.path.join(self.material_path,fname)))
		dataset=[cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)for img in dataset];dataset=np.array(dataset);variants=self.generate_variants(dataset);imgs1=dataset;imgs2=dataset;labels=[1]*len(dataset);randomized=np.array(dataset,copy=True);np.random.shuffle(randomized);assert np.sum(np.sum(randomized-dataset,axis=(0,1,2))==0)==0
		for k in range(len(variants)):one_var=variants[k];assert variants[k].shape==dataset.shape;imgs1=np.vstack((imgs1,one_var));imgs2=np.vstack((imgs2,dataset));labels+=[1]*len(dataset);imgs1=np.vstack((imgs1,one_var));imgs2=np.vstack((imgs2,randomized));labels+=[0]*len(one_var)
		np.savez_compressed(save_path,imgs1=imgs1,imgs2=imgs2,l=labels);return imgs1,imgs2,labels
	def replaceRandom(self,arr,num,val):temp=np.asarray(arr);shape=temp.shape;temp=temp.flatten();inds=np.random.choice(temp.size,size=num);temp[inds]=val;temp=temp.reshape(shape);return temp
	def generate_variants(self,data):
		positions=[(i,j)for i in range(-3,3)for j in range(-3,3)];positions.remove((0,0));ret=[]
		for pos in positions:tmp=np.roll(data,pos[0],axis=1);tmp=np.roll(tmp,pos[1],axis=2);ret.append(tmp)
		for intensity in range(1,5):tmp=np.where(data==255,255-intensity,data);ret.append(tmp);tmp=np.where(data==0,intensity,data);ret.append(tmp)
		for repeat in range(5):noise=np.random.normal(0,10,data.shape);tmp=data+noise;ret.append(tmp)
		for repeat in range(5):tmp=self.replaceRandom(data,10,0);ret.append(tmp);tmp=self.replaceRandom(data,10,255);ret.append(tmp)
		return ret
class TF_Models:
	@staticmethod
	def get_g_reference1():model=cifar_resnet20(block_type='original',shortcut_type='A',load_weights=True);return model
	def get_g_reference2():model=cifar_vgg11(load_weights=True);return model
	@staticmethod
	def mod2_same_cnn(input_shape=(32,32,3)):ly1_neurons=128;ly2_neurons=128;tmp_input=Input(shape=input_shape);x1=Conv2D(ly1_neurons,(3,3),padding='same')(tmp_input);x1=Activation('relu')(x1);x1=MaxPooling2D(pool_size=(2,2))(x1);x1=Dropout(.2)(x1);x1=Conv2D(ly2_neurons,(3,3),activation='relu')(x1);x1=Activation('relu')(x1);x1=MaxPooling2D(pool_size=(2,2))(x1);x1=Dropout(.2)(x1);x1=Flatten()(x1);shared_cnn_model=Model(tmp_input,x1,name='shared_cnn');img1=Input(shape=input_shape);img1_res=shared_cnn_model(img1);img2=Input(shape=input_shape);img2_res=shared_cnn_model(img2);x=keras.layers.concatenate([img1_res,img2_res]);x=Dense(128,activation='sigmoid')(x);x=Dense(32,activation='sigmoid')(x);x=Dense(1,activation='sigmoid')(x);model=Model([img1,img2],x,name='mod2_tf2');return model
def test_model(testX,testY,model,BS=128):
	predictions=model.predict(testX,batch_size=BS);predictions=(predictions>.5).astype(np.uint8).flatten();print(classification_report(testY,predictions,digits=4));images=[]
	for i in np.random.choice(np.arange(0,len(testY)),size=(49,)):
		prediction=model.predict([testX[0][(np.newaxis,i)],testX[1][(np.newaxis,i)]]);prediction=1 if prediction[0]>.5 else 0;img1=(testX[0][i]*255).astype('uint8');img2=(testX[1][i]*255).astype('uint8');img=np.vstack((img1,img2));assert img.shape==(64,32,3);color=0,255,0
		if prediction!=testY[i]:color=0,0,255
		cv2.putText(img,'{}'.format(prediction),(0,10),cv2.FONT_HERSHEY_SIMPLEX,.25,color,1);cv2.putText(img,'{}'.format(i),(0,25),cv2.FONT_HERSHEY_SIMPLEX,.25,color,1);images.append(img)
	montage=build_montages(images,(32,64),(7,7))[0];cv2.imwrite('g_baseline_results.png',montage)
def train(continue_training=False):
	is_tpu=False
	try:
		resolver=tf.distribute.cluster_resolver.TPUClusterResolver(tpu='test-tpu');tf.config.experimental_connect_to_cluster(resolver);tf.tpu.experimental.initialize_tpu_system(resolver);devs=tf.config.list_logical_devices('TPU');print('All devices: ',devs)
		if len(devs):is_tpu=True
		strategy=tf.distribute.TPUStrategy(resolver)
	except:pass
	handler=MyData();imgs1,imgs2,label1=handler.load_material();l=label1;assert len(imgs1)==len(imgs2)and len(imgs2)==len(l);imgs1=imgs1.astype('float32')/255.;imgs2=imgs2.astype('float32')/255.;trainX1,testX1,trainX2,testX2,trainY,testY=train_test_split(imgs1,imgs2,l,test_size=.3,stratify=l,random_state=42);del imgs1,imgs2,l;print('Train:{} Test:{}'.format(len(trainX1),len(testX1)));opt=keras.optimizers.Adam();model_name='g_baseline_model/model_v2.h5';model=None
	if is_tpu:
		with strategy.scope():
			if continue_training and os.path.isfile(model_name):model=load_model(model_name);print('\nContinue_training from {}'.format(model_name))
			else:model=TF_Models().mod2_same_cnn();model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'],steps_per_execution=50)
	elif continue_training and os.path.isfile(model_name):model=load_model(model_name);print('\nContinue_training from {}'.format(model_name))
	else:model=TF_Models().mod2_same_cnn();model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
	BS=128;EPOCHS=3;H=model.fit(x=[trainX1,trainX2],y=trainY,validation_data=([testX1,testX2],testY),steps_per_epoch=math.ceil(len(trainX1)/BS),epochs=EPOCHS,shuffle=True,verbose=1);model.save(model_name,save_format='h5');test_model([testX1,testX2],testY,model,BS=BS)
def test_using_text():model_name='g_baseline_model/model_v2.h5';model=load_model(model_name);data_handler=t_baseline.OurData();exp_img,exp,labels=data_handler.load_ocrb(label_collapse=False,get_new_data=False);ind=np.logical_and(exp==data_handler.single_char_names.index('S'),labels==1);exp_img=exp_img[ind].astype('uint8');tmp_loc_img,loc_exp,loc_labels=data_handler.processed_single_char_data();loc_labels=np.array(loc_labels);ind=np.logical_and(loc_exp==data_handler.single_char_names.index('S'),loc_labels==1);false_ind=np.logical_and(loc_exp==data_handler.single_char_names.index('S'),loc_labels==0);loc_img=tmp_loc_img[ind].astype('uint8');log_img_false=tmp_loc_img[false_ind].astype('uint8');exp_img=[cv2.cvtColor(cv2.bitwise_not(cv2.resize(image,(32,32))),cv2.COLOR_GRAY2BGR)for image in exp_img];exp_img=np.array(exp_img).astype('float32')/255.;loc_img=[cv2.cvtColor(cv2.bitwise_not(cv2.resize(image,(32,32))),cv2.COLOR_GRAY2BGR)for image in loc_img];loc_img=np.array(loc_img).astype('float32')/255.;log_img_false=[cv2.cvtColor(cv2.bitwise_not(cv2.resize(image,(32,32))),cv2.COLOR_GRAY2BGR)for image in log_img_false];log_img_false=np.array(log_img_false).astype('float32')/255.;code.interact(local=dict(locals(),**globals()));target_length=len(loc_img);exp=exp_img[np.newaxis,0,:];test_model([loc_img,np.broadcast_to(exp,(target_length,32,32,3))],[1]*target_length,model);target_length=len(log_img_false);test_model([log_img_false,np.broadcast_to(exp,(target_length,32,32,3))],[0]*target_length,model)
def robustness_evaluation_ref(model_path,write_output=False,dataset='icon',EPS=.3,NORM=np.inf):
	gpus=tf.config.list_physical_devices('GPU')
	for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
	test_acc_clean=tf.metrics.SparseCategoricalAccuracy();test_acc_cw2=tf.metrics.SparseCategoricalAccuracy();test_acc_cw22=tf.metrics.SparseCategoricalAccuracy();run_cw2=EPS==3 and NORM==2;t_class=t_baseline.OurModel();model=load_model(model_path);(_,_),(x_test,y_test)=cifar10.load_data();num_samples=120;x_test=x_test[:num_samples];y_test=y_test[:num_samples];assert len(x_test)==num_samples,len(x_test);assert len(y_test)==num_samples;x_test=tf.cast(x_test,tf.float32)/255.;y_pred=model(x_test);test_acc_clean(y_test,y_pred);from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2;x_cw2=[]
	if run_cw2:x_cw2=carlini_wagner_l2(model,x_test);y_pred_cw2=model(x_cw2);test_acc_cw2(y_test,y_pred_cw2);x_cw22=t_class.adjust(x_cw2,EPS,NORM,x_test);y_pred_cw22=model(x_cw22);test_acc_cw22(y_test,y_pred_cw22)
	if run_cw2:print('CW2: {:.2f} {:.2f}, 0s'.format(test_acc_cw2.result()*100,test_acc_cw22.result()*100))
def robustness_evaluation(model_path,write_output=False,dataset='icon',EPS=.3,NORM=np.inf):
	model=load_model(model_path);test_acc_clean=tf.metrics.BinaryAccuracy();test_acc_fgsm=tf.metrics.BinaryAccuracy();test_acc_pgd=tf.metrics.BinaryAccuracy();test_acc_cw2=tf.metrics.BinaryAccuracy();test_acc_mom=tf.metrics.BinaryAccuracy();test_acc_fgsm2=tf.metrics.BinaryAccuracy();test_acc_pgd2=tf.metrics.BinaryAccuracy();test_acc_cw22=tf.metrics.BinaryAccuracy();test_acc_mom2=tf.metrics.BinaryAccuracy();run_cw2=False;handler=MyData();attack_handler=t_baseline.OurAttacks();t_class=t_baseline.OurModel();assert dataset in['cifar','icon']
	if dataset=='cifar':imgs1,imgs2,labels=handler.load_ciphar(num=1)
	else:imgs1,imgs2,labels=handler.load_material()
	ind=labels==0;imgs1=imgs1[ind][:120];imgs2=imgs2[ind][:120];labels=labels[ind][:120];assert np.sum(labels==1)==0;imgs1=np.array(imgs1,dtype='float32')/255.;imgs2=np.array(imgs2,dtype='float32')/255.;labels=np.array(labels,dtype='uint8');y_pred=model([imgs1,imgs2]);test_acc_clean(labels,y_pred);x_fgm=attack_handler.fast_gradient_method(model,[imgs1,imgs2],EPS,NORM,labels,range_check=False);y_pred_fgm=model([x_fgm,imgs2]);test_acc_fgsm(labels,y_pred_fgm);x_fgm2=t_class.adjust(x_fgm,EPS,NORM,imgs1);y_pred_fgm2=model([x_fgm2,imgs2]);test_acc_fgsm2(labels,y_pred_fgm2);x_pgd=attack_handler.projected_gradient_descent(model,[imgs1,imgs2],EPS,EPS*.1,10,NORM,labels);y_pred_pgd=model([x_pgd,imgs2]);test_acc_pgd(labels,y_pred_pgd);x_pgd2=t_class.adjust(x_pgd,EPS,NORM,imgs1);y_pred_pgd2=model([x_pgd2,imgs2]);test_acc_pgd2(labels,y_pred_pgd2);x_mom=attack_handler.momentum(model,[imgs1,imgs2],eps=EPS,eps_iter=EPS*.05,nb_iter=10,norm=NORM);y_pred_mom=model([x_mom,imgs2]);test_acc_mom(labels,y_pred_mom);x_mom2=t_class.adjust(x_mom,EPS,NORM,imgs1);y_pred_mom2=model([x_mom2,imgs2]);test_acc_mom2(labels,y_pred_mom2);assert np.sum(imgs1<0)==0;assert np.sum(imgs1>1)==0;assert np.sum(imgs2<0)==0;assert np.sum(imgs2>1)==0;x_cw2=[];count_cw2=0
	if run_cw2:x_cw2=attack_handler.cw2(model,[imgs1,imgs2],y=labels);y_pred_cw2=model([x_cw2,imgs2]);test_acc_cw2(labels,y_pred_cw2);x_cw22=t_class.adjust(x_cw2,EPS,NORM,imgs1);y_pred_cw22=model([x_cw22,imgs2]);test_acc_cw22(labels,y_pred_cw22)
	print('Clean: {:.2f}'.format(test_acc_clean.result()*100));print('FGM: {:.2f} {:.2f}'.format(test_acc_fgsm.result()*100,test_acc_fgsm2.result()*100));print('PGD: {:.2f} {:.2f}'.format(test_acc_pgd.result()*100,test_acc_pgd2.result()*100));print('MOM: {:.2f} {:.2f}'.format(test_acc_mom.result()*100,test_acc_mom2.result()*100))
	if run_cw2:print('CW2: {:.2f} {:.2f}, 0s'.format(test_acc_cw2.result()*100,test_acc_cw22.result()*100))
	imgs2=(imgs2*255).astype('uint8')
	if write_output:
		adv_out_path='g_baseline_adv/{}_{}_{}_{}'.format(os.path.basename(model_path),dataset,EPS,NORM)
		if not os.path.isdir(adv_out_path):os.mkdir(adv_out_path)
		to_write_data=[imgs1,x_fgm,x_pgd,x_mom,x_cw2];names=['orig','fgm','pgd','mom','cw2']
		for i in range(len(to_write_data)):
			out_data=to_write_data[i]
			if not len(out_data):continue
			if i>0:keep=np.sum(out_data-imgs1,axis=(1,2,3))!=0
			out_dir=os.path.join(adv_out_path,names[i])
			if os.path.isdir(out_dir):shutil.rmtree(out_dir)
			os.mkdir(out_dir)
			for j in range(len(out_data)):
				if i>0 and not keep[j]:continue
				tmp=out_data[j];img2=imgs2[j]
				if type(tmp)!=np.ndarray:tmp=tmp.numpy()
				tmp=(tmp*255.).astype('uint8');cv2.imwrite('{}/{}.png'.format(out_dir,j),tmp);cv2.imwrite('{}/{}_img2.png'.format(out_dir,j),img2)
def test(model_path='g_baseline_model/model_v2.h5.copy',dataset='icon'):handler=MyData();imgs1,imgs2,labels=load_data(dataset,handler);model=load_model(model_path);test_model([imgs1,imgs2],labels,model,BS=128)
def load_data(dataset,handler):
	assert dataset in['cifar','icon']
	if dataset=='cifar':imgs1,imgs2,labels=handler.load_ciphar(num=1)
	else:imgs1,imgs2,labels=handler.load_material()
	imgs1=np.array(imgs1,dtype='float32')/255.;imgs2=np.array(imgs2,dtype='float32')/255.;labels=np.array(labels,dtype='uint8');return imgs1,imgs2,labels
def load_model(model_path):
	if model_path=='ref1':model=TF_Models.get_g_reference1()
	elif model_path=='ref2':model=TF_Models.get_g_reference2()
	else:model=keras.models.load_model(model_path)
	return model
def auto_attack(model_path,dataset='cifar',label_collapse=True,EPS=1,NORM=2,attacks_to_run=['fab','apgd-ce'],write_output=True,thresh=.5):
	from autoattack import utils_tf2,AutoAttack;handler=MyData();attack_handler=t_baseline.OurAttacks();t_class=t_baseline.OurModel();data,exp,labels=load_data(dataset,handler);ind=labels==0;data=data[ind][:120];exp=exp[ind][:120];labels=labels[ind][:120];assert np.sum(labels==1)==0;labels=np.array(labels);ind=labels==0;num_samples=120;data=np.array(data[ind][:num_samples]);orig=np.array(data,copy=True);exp=np.array(exp[ind][:num_samples]);labels=np.array(labels[ind][:num_samples]);import torch;data=torch.from_numpy(np.transpose(data,(0,3,1,2))).float().cuda();labels=torch.from_numpy(labels).float();gpus=tf.config.list_physical_devices('GPU')
	for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
	model=load_model(model_path);model_adapted=utils_tf2.ModelAdapter(model,num_classes=2);print('\n{} {} {} {}'.format(model_path,EPS,NORM,attacks_to_run));adversary=AutoAttack(model_adapted,norm='L{}'.format(NORM),eps=EPS,version='standard',is_tf_model=True,exp=exp,verbose=1,attacks_to_run=attacks_to_run,thresh=thresh);x_adv,ress=adversary.run_standard_evaluation_individual(data,labels,bs=num_samples);attack_results={}
	for(attack_name,advs)in x_adv.items():
		outputs=ress[attack_name][0].cpu().numpy()[:,1];np_x_adv=np.moveaxis(advs.cpu().numpy(),1,3);y_pred=model([np_x_adv,exp]);test_acc_unadjusted=tf.metrics.BinaryAccuracy(threshold=thresh);test_acc_unadjusted(labels,y_pred);adjusted=t_class.adjust(np_x_adv,EPS,NORM,data);test_acc_adjusted=tf.metrics.BinaryAccuracy(threshold=thresh);y_pred2=model([adjusted,exp]);test_acc_adjusted(labels,y_pred2);attack_results[attack_name]=test_acc_unadjusted.result()*100,test_acc_adjusted.result()*100
		if write_output:
			adv_out_path='g_baseline_adv/{}_{}_{}_{}/{}'.format(os.path.basename(model_path),dataset,EPS,NORM,attack_name)
			if os.path.isdir(adv_out_path):shutil.rmtree(adv_out_path)
			os.makedirs(adv_out_path)
			for j in range(len(np_x_adv)):
				tmp=np_x_adv[j];img2=orig[j]
				if type(tmp)!=np.ndarray:tmp=tmp.numpy()
				tmp=(tmp*255).astype(np.uint8);img2=(img2*255).astype(np.uint8);cv2.imwrite('{}/{}_{:.2f}.png'.format(adv_out_path,j,outputs[j]*100),cv2.bitwise_not(tmp));cv2.imwrite('{}/{}_img2.png'.format(adv_out_path,j),img2)
	for(attack_name,res)in attack_results.items():print('{} adjusted: {:.2f} {:.2f}'.format(attack_name,res[0],res[1]))
if __name__=='__main__':
	handler=MyData();imgs11,imgs21,l1=handler.load_material(load=True);code.interact(local=dict(locals(),**globals()));sys.exit(0)
	for mod in['g_baseline_model/ref1.h5']:robustness_evaluation_ref(mod,write_output=True,dataset='cifar',EPS=.1254,NORM=np.inf);robustness_evaluation_ref(mod,write_output=True,dataset='cifar',EPS=.2509,NORM=np.inf);robustness_evaluation_ref(mod,write_output=True,dataset='cifar',EPS=.5019,NORM=np.inf);robustness_evaluation_ref(mod,write_output=True,dataset='cifar',EPS=1,NORM=2);robustness_evaluation_ref(mod,write_output=True,dataset='cifar',EPS=2,NORM=2);robustness_evaluation_ref(mod,write_output=True,dataset='cifar',EPS=3,NORM=2)