'\nTake a screenshot of a webpage as input\nProduces \n    1. a segmented ground truth as output\n'
from PIL import Image
from io import BytesIO
import shutil,numpy as np,cv2,json,time,os,re,sys
from numpy.lib.utils import _median_nancheck
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import exceptions
from urllib.parse import urlparse
import tldextract,code
from bs4 import BeautifulSoup as BS
from bs4 import NavigableString
extra_text_classes={'www.jotform.com':['formFooter-text','form-subHeader','formFooter-button'],'Business Registration Form':['formFooter-text','form-subHeader','formFooter-button'],'PayPal: Send Money':[],'Free Texting Online | TextFree Web':['text','new-message-label','title','user-phone-number','new-message-to-label'],'#4 - oakland2021-mock':['feedback','pshint','field-d','js-edit-comment','revtime','x'],'Mail - Shawn Shuang - Outlook':['VgVT8LICPBt1qx8anfZFy','spFtLfql4yubeLFKiLvgx'],'(2) Submit to Reddit':['_3M6BmdyQcCEQZu-MylN14','_2N9ShiilNyzdd0B_i9geBj','_1sBmqB8geWKIW5Nt8svFgc'],'Feedback Form':['form-subHeader','formFooter-text','formFooter-button'],'Messenger':['_1htf','_1ht6','timestamp'],'shshuang@yahoo.com - Yahoo Mail':['P_oRhLS'],'ACH Authorization':['form-subHeader','formFooter-text','formFooter-button']}
def c(d):t=[len(v)for v in d.values()];return sum(t)/8
extra_text_tags={'(2) Submit to Reddit':['span','li'],'shshuang@yahoo.com - Yahoo Mail':['span'],'#4 - oakland2021-mock':['u']}
extra_input_classes={'www.jotform.com':[],'PayPal: Send Money':[],'Free Texting Online | TextFree Web':['emojionearea-editor','new-message-to-container'],'#4 - oakland2021-mock':[],'Mail - Shawn Shuang - Outlook':[],'(2) Submit to Reddit':[],'Feedback Form':[],'Messenger':[],'shshuang@yahoo.com - Yahoo Mail':['rte']}
extra_graphical_classes={'www.jotform.com':[],'PayPal: Send Money':['ppvx_icon___6-5-0'],'Free Texting Online | TextFree Web':['emojionearea-button-close','icon-close-icon','icon-pencil-icon','icon-refresh-icon'],'#4 - oakland2021-mock':[],'Mail - Shawn Shuang - Outlook':['ms-Icon--WaffleOffice365','root-61','ms-Icon--Help'],'(2) Submit to Reddit':['_1x6pySZ2CoUnAfsFhGe7J1'],'Feedback Form':[],'Messenger':['img']}
extra_graphical_style={'Messenger':['background-image']}
def draw_rects(img,locs,color,tags=[]):
	tmp=[loc[:4]for loc in locs];i=0
	for(x,y,w,h)in tmp:
		cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
		if i<len(tags):cv2.putText(img,str((tags[i],x,y,w,h)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
		i+=1
def no_more_white(res):return np.any(res!=0)
def find_largest_rect_area(orig,res):
	_,contours,_=cv2.findContours(res,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);contours=sorted(contours,key=cv2.contourArea,reverse=True);i=0
	for c in contours:x,y,w,h=cv2.boundingRect(c);res=cv2.rectangle(res,(x,y),(x+w,y+h),(255,0,0),1);cv2.imwrite('c_%s.png'%i,res);i+=1
	print('total contours %s'%len(contours));rect=cv2.boundingRect(contours[0]);return rect
def pil2bgr(pil):pil_image=pil.convert('RGB');open_cv_image=np.array(pil_image);return open_cv_image[:,:,::-1].copy()
def fill_the_rest(orig,page):
	'\n    A greedy method to draw rectangles. \n    ';res=page.copy();res=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY);_,res=cv2.threshold(res,170,255,cv2.THRESH_BINARY);w,h=res.shape;res=cv2.copyMakeBorder(res,h,h,w,w,cv2.BORDER_CONSTANT,None,0);rects=[];i=0
	while no_more_white(res):
		if i>20:sys.exit(1)
		rect=find_largest_rect_area(orig,res);rects.append(rect);draw_rects(res,[rect],0);i+=1
	return rects
TEXT=1
INPUT=2
GRAPHICS=3
def get_locations(lst,type_of_work=0):
	if type_of_work==TEXT:return[(int(ele.rect['x']),int(ele.rect['y']),int(ele.rect['width']),int(ele.rect['height']),ele.get_attribute('innerText'))for ele in lst]
	elif type_of_work==INPUT:return[(int(ele.rect['x']),int(ele.rect['y']),int(ele.rect['width']),int(ele.rect['height']),ele.get_attribute('data-component'))for ele in lst]
	return[(int(ele.rect['x']),int(ele.rect['y']),int(ele.rect['width']),int(ele.rect['height']))for ele in lst]
class VINT:
	def __init__(self,path,width=1920):
		self.path=path;self.html=None;self.counter=0;self.firefox=None;self.width=width;self.min_height=1080;self.input_rects=[];self.text_rects=[];self.graphical_rects=[];self.multi_rects={};self.site_name='';self.text_rects=[];self.graphical_rects=[];self.input_rects=[];self.static_sized=[];self.dynamic_sized=[];self.output_path='output';self.bs=None
		if len(path):self.update_exp_page(path)
		self.valid_char='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`!@#$%^&*()-=~_+[]\\{}|;\':",./<>?'
	def restart_firefox(self):
		if self.firefox is not None:self.firefox.quit()
		options=Options();options.headless=True;self.firefox=webdriver.Firefox(options=options);self.firefox.set_page_load_timeout(10)
	def update_exp_page(self,path):
		if self.counter>=300 or self.firefox is None:self.restart_firefox();self.counter=0
		if os.path.isfile(path):
			self.path=os.path.abspath(path);self.url='file:///'+path
			with open(self.path,'r')as inf:self.html=inf.read()
		elif path.startswith('<html>')or path.startswith('<!DOCTYPE html>'):self.url='data:text/html;charset=utf-8,{}'.format(path);self.html=path
		else:print('no support for live URL: %s'%path);sys.exit(1)
		try:self.orig,self.res=self.get_screenshot(self.url)
		except:print('self.orig:{}'.format(self.url));return False
		self.counter+=1;return True
	def generate_ground_truth(self):res,self.text_rects=self.detect_text(self.firefox,self.res);res,self.input_rects=self.detect_input_elements(self.firefox,self.res);res,self.graphical_rects=self.detect_graphical_elements(self.firefox,self.res);self.per_char_rects=self.get_per_char_location();cv2.imwrite('{}/after.png'.format(self.output_path),res);self.save_to_file()
	def do_work(self):
		try:self.generate_ground_truth()
		except Exception as e:exc_type,exc_obj,exc_tb=sys.exc_info();fname=os.path.split(exc_tb.tb_frame.f_code.co_filename)[1];print(e,exc_type,fname,exc_tb.tb_lineno);return False
		return True
	def get_screenshot(self,url,generate_ground_truth=False):
		try:self.firefox.get(url);default_w=self.width;default_h=self.min_height+74;self.firefox.set_window_size(default_w,default_h);total_height=self.firefox.execute_script('return document.body.parentNode.scrollHeight');self.firefox.set_window_size(default_w,max(total_height+74,default_h));page=self.selenium_screenshot(default_w,max(total_height+74,default_h))
		except Exception as e:print(e);return None,None
		orig=page.copy();return orig,page
	def deal_with_slideshows(self):
		slideshow_imgs=self.firefox.find_elements_by_css_selector('div[id="slideshow"]/img');img_sources=[x.get_attribute('src')for x in slideshow_imgs];locs=get_locations(slideshow_imgs);pos=0
		for(x,y,w,h)in locs:
			if sum(x,y,w,h)!=0 and pos==0:pos=x,y,w,h
			elif sum(x,y,w,h)==0 and pos!=0:print('slide show error')
		assert pos in self.graphical_rects,'slide show error';self.graphical_rects.remove(pos);self.multi_rects[pos]=img_sources
	def deal_with_hover(self):0
	def is_submit_present(self):
		try:self.firefox.find_element_by_css_selector('button[type="submit"]');return True
		except:return False
	def fix_radio_listbox(self,elements):
		for ele in elements:
			code_val=ele.get('value')
			if not ele.next_sibling:display_val=ele.parent.next_sibling if isinstance(ele.parent.next_sibling,NavigableString)else ele.parent.next_sibling.get_text()
			else:display_val=ele.next_sibling.next_sibling if isinstance(ele.next_sibling.next_sibling,NavigableString)else ele.next_sibling.next_sibling.get_text()
			if code_val!=display_val:print('Must fix code val and display val: (%s, %s)'%(code_val,display_val))
	def treat_inputs_types(self):
		radios=[x for x in self.bs.find_all('input',type='radio')];self.fix_radio_listbox(radios);checkboxes=[x for x in self.bs.find_all('input',type='checkbox')];self.fix_radio_listbox(checkboxes);options=self.bs.find_all('option')
		for opt in options:print('Must fix listbox code and display val: (%s,%s)'%(opt.get('value'),opt.get_text()))
	def add_css_reset(self):
		head=self.bs.find('head');insert_pos=2
		if head.contents[1].name!='meta':insert_pos=1
		head.insert(insert_pos,'<link rel="stylesheet" type="text/css" href="https://necolas.github.io/normalize.css/8.0.1/normalize.css">')
	def validate_page(self):
		'\n        returns the dynmaic elements in \n        '
		if self.bs is None:self.bs=BS(self.html,'lxml')
		if not self.is_submit_present():print('No submit button found')
		videos=self.bs.find_all('video')
		if videos:print('video not supported')
		iframes=self.bs.find_all('iframe')
		if iframes:print('iframes not supported')
		self.treat_inputs_types();file_inputs=self.bs.find_all('input',type='file')
		if file_inputs:print('file input not supported')
		if'caret-color'in self.html:print('custermized caret-color is not supported')
		pattern=re.search(':\\s*focus[^\\{]*\\{[^\\}]*outline',self.html)
		if pattern:print('custermized focus box is not supported')
	def add_extra_class(self,type_dict,type_of_work=0):
		ret=[]
		if self.site_name in type_dict:
			for cl in type_dict[self.site_name]:elements=self.firefox.find_elements_by_xpath("//*[contains(@class, '%s')]"%cl);ret+=get_locations(elements,type_of_work)
		return ret
	def add_extra_tags(self,type_dict,type_of_work):
		ret=[]
		if self.site_name in type_dict:
			for tn in type_dict[self.site_name]:elements=self.firefox.find_elements_by_tag_name(tn);ret+=get_locations(elements,type_of_work)
		return ret
	def add_extra_style(self,type_dict):
		ret=[]
		if self.site_name in type_dict:
			for sty in type_dict[self.site_name]:ret+=get_locations(self.firefox.find_elements_by_xpath("//*[contains(@style, '%s')]"%sty))
		return ret
	def count_valid_char(self,string):
		if not string:return 0
		return len([char for char in string if char.isalpha()])
	def get_per_char_location(self):js_code="\n        function deal_with_one_ele(node, range, array) {\n            if (node.childNodes.length != 0) {\n                for (let  i = 0; i < node.childNodes.length; i++) {\n                    deal_with_one_ele(node.childNodes[i], range, array);\n                }\n            }\n            for (let i =0; i < node.length; i++) {\n                char = node.textContent.substring(i, i+1);\n                range.setStart(node, i);\n                range.setEnd(node, i+1);\n                rect = range.getBoundingClientRect();\n                if (char) {\n                    array.push([rect.x, rect.y, rect.width, rect.height, char]);\n                }\n            }\n        }\n\n        var array = [];\n        const range = document.createRange();\n        ele_types = ['p', 'label', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong'];\n        for (let i = 0;  i < ele_types.length; i++) {\n            ele_type = ele_types[i];\n            all_eles = document.getElementsByTagName(ele_type);\n            for (let j = 0; j < all_eles.length; j++) {\n                ele = all_eles[j];\n                deal_with_one_ele(ele, range, array);\n            }\n        }\n        return array;\n        ";ret=self.firefox.execute_script(js_code);ret=[(x,y,w,h,char)for(x,y,w,h,char)in ret if char and char in self.valid_char and w];return ret
	def detect_text(self,firefox,res):all_eles=[x for x in firefox.find_elements(By.XPATH,'//*[not(*)]')if x.text is not None and len(x.text)>0 and str(x.text)!='N/A'and self.count_valid_char(x.text)>0 and x.tag_name not in['select','option']and x.is_displayed()];type_of_work=TEXT;locs=get_locations(all_eles,type_of_work);locs+=self.add_extra_class(extra_text_classes,type_of_work);locs+=self.add_extra_tags(extra_text_tags,type_of_work);draw_rects(res,locs,(0,255,0),tags=[x.tag_name for x in all_eles]);return res,locs
	def detect_graphical_elements(self,firefox,res):
		try:imgs=firefox.find_elements_by_tag_name('img');svgs=firefox.find_elements_by_tag_name('svg')
		except exceptions.StaleElementReferenceException:imgs=[];svgs=[]
		locs=get_locations(imgs+svgs);locs+=self.add_extra_class(extra_graphical_classes);locs+=self.add_extra_style(extra_graphical_style);draw_rects(res,locs,(0,0,255));return res,locs
	def remove_white_border(self,img):gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);blur=cv2.GaussianBlur(gray,(25,25),0);thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1];noise_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3));opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,noise_kernel,iterations=2);close_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7));close=cv2.morphologyEx(opening,cv2.MORPH_CLOSE,close_kernel,iterations=3);coords=cv2.findNonZero(close);x,y,w,h=cv2.boundingRect(coords);cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),2);crop=img[y:y+h,x:x+w];return crop
	def generate_dynamic_ground_truth(self,ele):import pyautogui;pyautogui.press('f11');im1=pil2bgr(pyautogui.screenshot());self.firefox.set_window_position(0,0);x,y,w,h=get_locations([ele])[0];pyautogui.pause=2;pyautogui.click(x=x+2,y=y+2);im2=pil2bgr(pyautogui.screenshot());diff=cv2.subtract(im1,im2);Conv_hsv_Gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY);ret,mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU);im2[mask==255]=[0,0,0];return self.remove_white_border(im2)
	def detect_dynamically_sized(self,firefox,res):
		eles=firefox.find_elements_by_tag_name('select');dropdowns=[ele for ele in eles if ele.get_attribute('size')=='0'];locs=get_locations(dropdowns);draw_rects(res,locs,(255,0,255))
		for i in range(len(locs)):ele=dropdowns[i];appearance=self.generate_dynamic_ground_truth(ele);save_path=self.save_recursive(appearance);locs[i]=locs[i][0],locs[i][1],locs[i][2],locs[i][3],save_path
		return res,locs
	def selenium_screenshot(self,w,h):page=self.firefox.get_screenshot_as_png();page=Image.open(BytesIO(page));page=page.crop((0,0,w,h-74));return cv2.cvtColor(np.array(page),cv2.COLOR_RGB2BGR)
	def generate_select_ground_truth(self,ele):old_size=ele.get_attribute('size');select=Select(ele);num_options=len(select.options);assert ele.get_attribute('id');self.firefox.execute_script("document.getElementById('%s').size=%s"%(ele.get_attribute('id'),num_options));ele_x,ele_y,ele_w,ele_h=get_locations([ele])[0];self.firefox.execute_script("document.getElementById('%s').size=%s"%(ele.get_attribute('id'),old_size));return self.page[ele_y:ele_y+ele_h,ele_x:ele_x+ele_w]
	def update_out_path(self,new_path):self.output_path=new_path
	def save_recursive(self,appearance):
		save_path='output/%s'%time.time();os.mkdir(save_path);cv2.imwrite('%s/orig.png'%save_path,appearance);h,w,_=appearance.shape;big={'graphic':[[0,0,w,h]]}
		with open('%s/rects.txt'%save_path,'w')as outf:outf.write(json.dumps(big))
		return save_path
	def detect_static_sized(self,firefox,res):
		eles=firefox.find_elements_by_tag_name('select');selects=[ele for ele in eles if ele.get_attribute('size')!='0'];locs=get_locations(selects);draw_rects(res,locs,(0,255,255));h,w,_=self.res.shape
		for i in range(len(locs)):ele=selects[i];appearance=self.generate_select_ground_truth(ele);save_path=self.save_recursive(appearance);locs[i]=locs[i][0],locs[i][1],max(locs[i][2],w),max(locs[i][3],h),save_path
		return res,locs
	def detect_input_elements(self,firefox,res):inputs=firefox.find_elements_by_tag_name('input');text_areas=firefox.find_elements_by_tag_name('textarea');eles=firefox.find_elements_by_tag_name('select');dropdowns=[ele for ele in eles if ele.get_attribute('size')=='0'];locs=get_locations(inputs+text_areas+dropdowns,type_of_work=INPUT);locs+=self.add_extra_class(extra_input_classes);draw_rects(res,locs,(255,0,0));return res,locs
	def save_to_file(self):
		cv2.imwrite('{}/orig.png'.format(self.output_path),self.orig);self.text_rects=[x for x in self.text_rects if sum(x[:4])!=0];self.graphical_rects=[x for x in self.graphical_rects if sum(x[:4])!=0];self.input_rects=[x for x in self.input_rects if sum(x[:4])!=0];self.static_sized=[x for x in self.static_sized if sum(x[:4])!=0];self.dynamic_sized=[x for x in self.dynamic_sized if sum(x[:4])!=0];big={'text':self.text_rects,'graphic':self.graphical_rects,'input':self.input_rects,'static_sized':self.static_sized,'dynamic_sized':self.dynamic_sized,'per_char':self.per_char_rects}
		with open('{}/rects.txt'.format(self.output_path),'w')as file:file.write(json.dumps(big))
if __name__=='__main__':
	sys.argv.append('/home/ss/Documents/VT/end-to-end/page/Business Registration Form.html')
	if os.path.isdir('output'):shutil.rmtree('output')
	os.mkdir('output');vint=VINT(sys.argv[1]);vint.do_work()