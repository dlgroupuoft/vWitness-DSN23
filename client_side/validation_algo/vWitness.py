from os.path import isfile
import new_algo as na,cv2,sys,json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
import numpy as np,os,code,imutils,time,test_colorio as tc,math,invoke_baselines as models,hashlib
from new_algo import Cache
class Validator:
	def __init__(self,loc,rects_f='output/rects.txt',exp_f='output/orig.png',fname='',use_g_model=True,use_t_model=True,write_failed=False,use_cache=False,g_model_path='g_baseline_model/model_v2.h5'):
		self.g_model_path=g_model_path;self.width=1920;self.min_height=1080;self.update_loc(loc);self.update_ground_truth(rects_f,exp_f);self.input_values={};self.frame_rects={};self.success_rects={'text':[],'graphics':[]};self.gv=tc.GValidator();self.models=None;self.debug=0;self.use_g_model=use_g_model;self.use_t_model=use_t_model;self.write_failed=write_failed;self.fname=fname;self.use_cache=use_cache;self.f_cache,self.t_cache,self.g_cache=None,None,None
		if self.use_cache:self.f_cache,self.t_cache,self.g_cache=Cache(),Cache(),Cache()
	def update_ground_truth(self,rects_f,exp_f):
		self.exp=cv2.imread(exp_f);self.exp_full=self.exp
		with open(rects_f,'r')as file:data=file.read();self.rects=json.loads(data)
	def update_loc(self,loc):
		if isinstance(loc,(np.ndarray,np.generic)):self.loc=loc
		elif isinstance(loc,str)and os.path.isfile(loc)and loc.endswith('html'):loc='file:///'+os.path.abspath(loc);self.loc,self.firefox=self.get_screenshot(loc)
		elif isinstance(loc,str)and os.path.isfile(loc)and(loc.endswith('png')or loc.endswith('jpg')):self.loc=cv2.imread(loc);cv2.cvtColor(self.loc,cv2.COLOR_BGRA2BGR)
		elif isinstance(loc,str)and loc.startswith('data:text/html;charset=utf-8'):self.loc,self.firefox=self.get_screenshot(loc)
		else:print('error in loc format:{}'.format(loc[:20]))
	def get_screenshot(self,url):options=Options();options.headless=True;firefox=webdriver.Firefox(options=options);default_width=self.width;firefox.get(url);total_height=firefox.execute_script('return document.body.parentNode.scrollHeight');firefox.set_window_size(default_width,total_height);page=firefox.get_screenshot_as_png();page=cv2.imdecode(np.frombuffer(page,np.uint8),-1);page=cv2.cvtColor(page,cv2.COLOR_BGRA2BGR);orig=page.copy();return orig,firefox
	def is_valid_rect(self,x,y,w,h):return w*h>10
	def align_text(self,exp_t,x,y,w,h):
		adjust_height=self.min_height//70;adjust_width=self.width//70;start_y=max(y-adjust_height,0);start_x=max(0,x-adjust_width);end_y=min(y+h+adjust_height,self.min_height);end_x=min(x+w+adjust_width,self.width);new_loc=self.loc[start_y:end_y,start_x:end_x].copy();rects=na.text_detect_run(new_loc)
		if not rects:return-1
		new_x=rects[0][0]+start_x;new_y=rects[0][1]+start_y;new_w=min(rects[0][2],w);new_h=min(rects[0][3],h)
		if self.debug:cv2.rectangle(new_loc,rects[0][:2],(rects[0][0]+new_w,rects[0][1]+new_h),(0,0,0),2);cv2.imwrite('tmp/tmp_loc_{}_v299.png'.format(''.join(x for x in exp_t if x.isalnum())),new_loc)
		return new_x,new_y,new_w,new_h
	def intersects(self,p1,p2):
		p1_tr=p1[0];p2_tr=p2[0];p1_bl=p1[1];p2_bl=p2[1]
		if not(p1_tr[0]<p2_bl[0]or p1_bl[0]>p2_tr[0]or p1_tr[1]<p2_bl[1]or p1_bl[1]>p2_tr[1]):return 0
		tr=max(p1_tr[0],p2_tr[0]),max(p1_tr[1],p2_tr[1]);bl=min(p1_tr[0],p2_bl[0]),min(p1_bl[0],p2_bl[1]);return tr,bl
	def check_overlappings(self,loc,p1_tr,p1_bl,id):
		loc_before=loc;overlapped=False
		for(x,y,w,h,_)in self.success_rects['text']+self.success_rects['graphics']:
			tmp_tr=x,y;tmp_bl=x+w,y+h;res=self.intersects((p1_tr,p1_bl),(tmp_tr,tmp_bl))
			if res!=0:overlap_tr,overlap_bl=res;assert len(overlap_bl)==2 and len(overlap_tr)==2;overlap_tr=overlap_tr[0]-p1_tr[0],overlap_tr[1]-p1_tr[1];overlap_bl=overlap_bl[0]-p1_bl[0],overlap_bl[1]-p1_bl[1];cv2.rectangle(loc,overlap_tr,overlap_bl,(255,255,255),thickness=-1);overlapped=True
		if overlapped and self.debug:cv2.imwrite('tmp/{}_before.png'.format(id),loc_before);cv2.imwrite('tmp/{}_after.png'.format(id),loc)
		return loc
	def compare_graphics_model(self,annotation_path='',failed_path=''):
		if not self.models:self.models=models.Model_Validator(g_model_path=self.g_model_path)
		if self.loc is None or self.exp is None:print('Error ');return True
		ret=True;locs=[];exps=[]
		for(x,y,w,h)in self.frame_rects['graphic']:
			if not self.is_valid_rect(x,y,w,h):continue
			locs.append(self.loc[y:y+h,x:x+w]);exps.append(self.exp[y:y+h,x:x+w])
		if not(len(locs)and len(exps)):print('Error, no locs or exps');code.interact(local=dict(locals(),**globals()))
		ret=ret and self.models.validate_g(locs,exps,annotation_path,failed_path);return ret
	def compare_text_model(self):
		if not self.models:self.models=models.Model_Validator(g_model_path=self.g_model_path)
		ret=True;i=0;locs=[];exps=[];rect=self.frame_rects['per_char']if'per_char'in self.frame_rects else self.frame_rects['text'];grayed=cv2.cvtColor(self.loc,cv2.COLOR_BGR2GRAY);h,w=grayed.shape;grayed_resized=cv2.resize(grayed,None,fx=10,fy=10);res=[-1]*len(rect)
		for(i,(x,y,w,h,exp_t))in enumerate(rect):
			x,y,w,h=round(x*10),round(y*10),round(w*10),round(h*10);region_of_interest=grayed_resized[y:y+h,x:x+w]
			if self.use_cache and self.t_cache.hit_img(region_of_interest):res[i]=self.t_cache.get_img(region_of_interest)
			else:locs.append(region_of_interest);exps.append(exp_t)
		validate_res=self.models.validate_t(locs,exps,page_name=self.fname,write_failed=self.write_failed);validate_res_index=0
		for(i,(x,y,w,h,exp_t))in enumerate(rect):
			x,y,w,h=round(x*10),round(y*10),round(w*10),round(h*10);region_of_interest=grayed_resized[y:y+h,x:x+w]
			if res[i]==-1:
				res[i]=validate_res[validate_res_index];validate_res_index+=1
				if self.use_cache:self.t_cache.update_img(region_of_interest,res[i])
		assert-1 not in res,res;return sum(res)==len(rect)
	def compare_text(self):
		ret=True;i=0
		for(x,y,w,h,exp_t)in self.frame_rects['text']:
			loc=self.loc[y:y+h,x:x+w].copy();old_loc=loc;i+=1;loc_v1=None;local_t,loc_conf='',0;comparison_res=False
			if len(exp_t)<=3:
				align_rects=self.align_text(exp_t,x,y,w,h)
				if align_rects!=-1:new_x,new_y,new_w,new_h=align_rects;new_x,new_y,new_w,new_h=new_x,new_y,int(new_w*.6),new_h
				else:width=10*len(exp_t);new_x,new_y,new_w,new_h=x,y,20,h
				single_loc=self.loc[new_y:new_y+new_h,new_x:new_x+new_w].copy();loc=np.concatenate((single_loc,single_loc),axis=1);loc=np.concatenate((loc,single_loc),axis=1);loc=np.concatenate((loc,single_loc),axis=1);loc=np.concatenate((loc,single_loc),axis=1);local_t,loc_conf=na.ocr_run(loc);local_t=local_t.replace(' ','');local_t=local_t[0:int(len(local_t)/5)];comparison_res=na.text_comparison(local_t,exp_t)
				if not comparison_res:
					if self.debug:exp_t=''.join(x for x in exp_t if x.isalnum());cv2.imwrite('tmp/tmp_loc_{}_v1.png'.format(exp_t),loc)
					print('loc:%s exp:%s'%(local_t.replace('\n',''),exp_t.replace('\n','')))
			else:
				local_t,loc_conf=na.ocr_run(loc);comparison_res=na.text_comparison(local_t,exp_t)
				if comparison_res:self.success_rects['text'].append((x,y,w,h,exp_t))
				exp_t=exp_t.encode('ascii','ignore').decode()
				if not comparison_res:
					local_t2,local_t3='','';adjust_x,adjust_h=5,10;start_x,start_y=max(0,x-adjust_x),max(0,y-adjust_h);loc=self.loc[start_y:y+h+adjust_h,start_x:x+w+adjust_x].copy();loc=self.check_overlappings(loc,(start_x,start_y),(x+w+adjust_x,y+h+adjust_h),exp_t);loc_v1=loc;local_t2,loc_conf=na.ocr_run(loc);comparison_res=na.text_comparison(local_t2,exp_t)
					if comparison_res:self.success_rects['text'].append((start_x,start_y,w+adjust_x,h+adjust_h,exp_t))
					if not comparison_res:
						align_rects=self.align_text(exp_t,x,y,w,h)
						if align_rects!=-1:
							new_x,new_y,new_w,new_h=align_rects;loc=self.loc[new_y:new_y+new_h,new_x:new_x+new_w].copy();local_t3,loc_conf=na.ocr_run(loc);comparison_res=na.text_comparison(local_t3,exp_t)
							if comparison_res:self.success_rects['text'].append((new_x,new_y,new_w,new_h,exp_t))
						else:print('No text detected in v3')
				if not comparison_res:
					if self.debug:exp_t=''.join(x for x in exp_t if x.isalnum());fname=exp_t[:10];cv2.imwrite('tmp/tmp_loc_{}_v1.png'.format(fname),old_loc);cv2.imwrite('tmp/tmp_loc_{}_v2.png'.format(fname),loc_v1);cv2.imwrite('tmp/tmp_loc_{}_v3.png'.format(fname),loc)
					print('loc:%s\nloc2:%s\\mloc3:%s\nexp:%s'%(local_t.replace('\n',' '),local_t2.replace('\n',' '),local_t3.replace('\n',' '),exp_t.replace('\n',' ')))
			ret=ret and comparison_res
		return ret
	def color_comparison2(self,loc,exp,prefix):res=self.gv.tiered_color_comparison(loc,exp);return sum(res)==0
	def color_comparison(self,loc,exp,prefix):
		loc_h,loc_w,_=loc.shape;res=na.color_comparison(loc,exp)
		if self.debug:cv2.imwrite('tmp/{}_loc.png'.format(prefix),loc);cv2.imwrite('tmp/{}_exp.png'.format(prefix),loc);print(res,loc_h*loc_w)
		return max(res[:3])/(loc_h*loc_w)<.01
	def align_graphics(self,exp,x,y,w,h,i):exp_w,exp_h,_=exp.shape;adjust_height=self.min_height//70;adjust_width=self.width//70;start_y=max(y-adjust_height,0);start_x=max(0,x-adjust_width);new_loc=self.loc[start_y:min(y+h+adjust_height,self.min_height),start_x:min(x+w+adjust_width,self.width)].copy();res=cv2.matchTemplate(new_loc,exp,cv2.TM_CCOEFF);min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res);top_left=max_loc;bottom_right=top_left[0]+w,top_left[1]+h;top_left=top_left[0]+start_x,top_left[1]+start_y;return top_left
	def compare_graphics(self):
		ret=True;i=0
		for(x,y,w,h)in self.frame_rects['graphic']:
			if not self.is_valid_rect(x,y,w,h):continue
			loc=self.loc[y:y+h,x:x+w].copy();exp=self.exp[y:y+h,x:x+w].copy();tmp=False;tmp=self.color_comparison2(loc,exp,i)
			if not tmp:
				old_loc=loc.copy();new_x,new_y=self.align_graphics(exp,x,y,w,h,i);loc=self.loc[new_y:new_y+h,new_x:new_x+w].copy();tmp=self.color_comparison2(loc,exp,'{}_second_try'.format(i))
				if tmp:self.success_rects['graphics'].append((new_x,new_y,w+new_x,h+new_x,''))
				if not tmp and self.debug:cv2.imwrite('tmp/g%s_loc_v1.png'%i,old_loc);cv2.imwrite('tmp/g%s_loc_v2.png'%i,loc);cv2.imwrite('tmp/g%s_exp.png'%i,exp)
			ret=ret and tmp;i+=1
		return ret
	def compare_static_sized(self):
		ret=True
		for(x,y,w,h,path)in self.frame_rects['static_sized']:
			if not self.is_valid_rect(x,y,w,h):continue
			sub=Validator(self.loc[y:y+h,x:x+w].copy(),os.path.join(path,'rects.txt'),os.path.join(path,'orig.png'));ret=ret and sub.do_work()
		return ret
	def multi_scale_template_match(self):
		template=cv2.cvtColor(self.loc,cv2.COLOR_BGR2GRAY);template=cv2.Canny(template,50,200);tH,tW=template.shape[:2];image=self.loc;gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);found=None
		for scale in np.linspace(.2,1.,20)[::-1]:
			resized=imutils.resize(gray,width=int(gray.shape[1]*scale));r=gray.shape[1]/float(resized.shape[1])
			if resized.shape[0]<tH or resized.shape[1]<tW:break
			edged=cv2.Canny(resized,50,200);result=cv2.matchTemplate(edged,template,cv2.TM_CCOEFF);_,maxVal,_,maxLoc=cv2.minMaxLoc(result)
			if found is None or maxVal>found[0]:found=maxVal,maxLoc,r
		ret=None;max_val=0
		if not found is None:_,maxLoc,r=found;startX,startY=int(maxLoc[0]*r),int(maxLoc[1]*r);endX,endY=int((maxLoc[0]+tW)*r),int((maxLoc[1]+tH)*r);ret=startX,endX,startY,endY;max_val=found[0]
		return ret,max_val
	def reverse_validation(self):
		found,max_val=self.multi_scale_template_match();print(max_val,found);h1,w1,_=self.loc.shape;h2,w2,_=self.exp.shape
		if w1<=w2 or h1<=h2:return False
		ret=True
		if found is not None:x1,x2,y1,y2=found;self.loc=self.loc[y1:min(y2,y1+h2),x1:min(x2,x1+w2)];ret=self.do_work()
		return ret
	def compare_dynamic_sized(self):
		ret=True
		for(x,y,w,h,path)in self.frame_rects['dynamic_sized']:
			if not self.is_valid_rect(x,y,w,h):continue
			sub=Validator(self.loc[y:y+h,x:x+w].copy(),os.path.join(path,'rects.txt'),os.path.join(path,'orig.png'));ret=ret and sub.reverse_validation()
		return ret
	def update_frame_rects(self,x1,y1,x2,y2,coor=None):
		'\n\t\tx1, y2, x2, y2 is a region on self.exp that is the current focus\n\t\t'
		for k in self.rects.keys():self.frame_rects[k]=[]
		if coor is not None:x1,x2,y1,y2=coor
		self.frame_rects['text']=[(x-x1,y-y1,w,h,exp_t)for(x,y,w,h,exp_t)in self.rects['text']if x>=x1 and y>=y1 and x+w<=x2 and y+h<=y2]
		if'per_char'in self.rects:self.frame_rects['per_char']=[(x-x1,y-y1,w,h,exp_t)for(x,y,w,h,exp_t)in self.rects['per_char']if x>=x1 and y>=y1 and x+w<=x2 and y+h<=y2]
		self.frame_rects['graphic']=[(x-x1,y-y1,w,h)for(x,y,w,h)in self.rects['graphic']if x>=x1 and y>=y1 and x+w<=x2 and y+h<=y2];self.frame_rects['input']=[(x-x1,y-y1,w,h,exp_t)for(x,y,w,h,exp_t)in self.rects['input']if x>=x1 and y>=y1 and x+w<=x2 and y+h<=y2];self.frame_rects['static_sized']=[(x-x1,y-y1,w,h)for(x,y,w,h)in self.rects['static_sized']if x>=x1 and y>=y1 and x+w<=x2 and y+h<=y2];self.frame_rects['dynamic_sized']=[(x-x1,y-y1,w,h)for(x,y,w,h)in self.rects['dynamic_sized']if x>=x1 and y>=y1 and x+w<=x2 and y+h<=y2];self.boundary_elements=[(x-x1,y-y1,min(self.width-x,w),min(self.min_height-y,h))for(x,y,w,h)in self.rects['graphic']+self.rects['static_sized']+self.rects['dynamic_sized']if x>=x1 and y>=y1 and x<=x2 and y<=y2 and(x+w-x1>x2 or y+h-y1>y2)];self.boundary_elements+=[(x-x1,y-y1,min(self.width-x,w),min(self.min_height-y,h))for(x,y,w,h,_)in self.rects['text']+self.rects['input']if x>=x1 and y>=y1 and x<x2 and y<y2 and(x+w-x1>x2 or y+h-y1>y2)]
		if self.debug:
			loc_tmp=self.loc.copy()
			for(x,y,w,h)in self.boundary_elements:cv2.rectangle(loc_tmp,(x,y),(x+w,y+h),(0,0,0),thickness=-1)
			cv2.imwrite('tmp/loc_tmp.png',loc_tmp)
	def extract_input(self):
		for(x,y,w,h,label)in self.frame_rects['input']:
			text_val,conf=na.ocr_run(self.loc[y:y+h,x:x+w])
			if conf>80 and text_val:self.input_values[label]=text_val.strip()
		if self.input_values:print('Extracted inputs are: %s'%self.input_values)
	def check_position_constraints(self,x,y,new_x,new_y):return abs(x-new_x)/70>self.width or abs(y-new_y)/70>self.height
	def validate_positions(self):
		'\n\t\tIs this really necessary ?? \n\t\tBecause text detection and image realignment already only does within the position of the \n\t\t';text_label_to_pos={}
		for(x,y,_,_,exp_t)in self.frame_rects['text']:text_label_to_pos[exp_t]=x,y
		for(new_x,new_y,label)in self.updated_text_pos:
			if self.check_position_constraints(x,y,new_x,new_y):print('failed positioning constraint')
		for(new_x,new_y,label)in self.updated_graphics_pos:
			if self.check_position_constraints(x,y,new_x,new_y):print('failed positioning constraint')
	def do_work(self,coor=None,annotation_path='',failed_path=''):
		if self.use_cache and self.f_cache.hit_img(self.loc):return self.f_cache.get_img(self.loc)
		if self.loc.shape!=self.exp.shape:return False
		res=cv2.matchTemplate(self.exp_full,self.loc,cv2.TM_CCOEFF);min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res);x1,y1=max_loc;loc_h,loc_w,_=self.loc.shape;x2,y2=x1+loc_w,y1+loc_h;self.exp=self.exp_full[y1:y2,x1:x2];self.update_frame_rects(x1,y1,x2,y2,coor)
		if self.use_t_model:text_succ=self.compare_text_model()if'text'in self.frame_rects else True
		else:text_succ=self.compare_text()if'text'in self.frame_rects else True
		if self.use_g_model:graphical_succ=self.compare_graphics_model(annotation_path,failed_path)if'graphic'in self.frame_rects else True
		else:graphical_succ=self.compare_graphics()if'graphic'in self.frame_rects else True
		static_sized_succ=self.compare_static_sized()if'static_sized'in self.frame_rects else True;dynamic_sized_succ=self.compare_dynamic_sized()if'dynamic_sized'in self.frame_rects else True
		if self.use_cache:self.f_cache.update_img(self.loc,text_succ and graphical_succ and static_sized_succ and dynamic_sized_succ)
		return text_succ and graphical_succ and static_sized_succ and dynamic_sized_succ
if __name__=='__main__':val=Validator();print('Validation is %s'%val.do_work())