'\nTake a screenshot of a webpage as input\nProduces \n    1. a segmented ground truth as output\n'
import shutil,numpy as np,cv2,json,time,os,cssutils,re,sys
from numpy.lib.utils import _median_nancheck
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selectolax.parser import HTMLParser
from selenium.common import exceptions
from urllib.parse import urlparse
import tldextract,code
from bs4 import BeautifulSoup as BS
from bs4 import NavigableString
class VINT:
	def __init__(self,path='',width=1920):
		self.path=path;self.html=None;self.bs=None
		if len(path):self.update_exp_page(path)
		self.errored=set();self.minirest=HTMLParser('<link href="sanitize.css" rel="stylesheet" type="text/css">').css_first('link');self.font_css=HTMLParser('<link href="https://fonts.googleapis.com/css?family=Lato&display=swap" rel="stylesheet">').css_first('link');self.fonts=HTMLParser('<style>\n            @font-face {font-family: "ocrb";src: url("fonts/OcrB2.ttf");}\n            @font-face {font-family: "arial-light";src: url("fonts/ARIALLGT.TTF");}\n            @font-face {font-family: "arial-black";src: url("fonts/ARIBLK.TTF");}\n            @font-face {font-family: "encodesans";src: url("fonts/EncodeSans/EncodeSans-Regular.ttf");}\n            @font-face {font-family: "encodesans-black";src: url("fonts/EncodeSans/EncodeSans-Black.ttf");}\n            @font-face {font-family: "encodesans-light";src: url("fonts/EncodeSans/EncodeSans-ExtraLight.ttf");}\n            @font-face {font-family: "encodesans-condensed";src: url("fonts/EncodeSans_Condensed/EncodeSans_Condensed-Regular.ttf");}\n            @font-face {font-family: "encodesans-expanded";src: url("fonts/EncodeSans_Expanded/EncodeSans_Expanded-Regular.ttf");}\n            @font-face {font-family: "ocrb";src: url("fonts/RobotoMono-Regular.ttf");}</style>\n        ').css_first('style')
	def update_exp_page(self,path):
		if os.path.isfile(path)and path.endswith('html'):
			self.path=os.path.abspath(path);self.url='file:///'+path
			with open(path,'r')as inf:self.html=inf.read()
		elif path.startswith('<html>')or path.startswith('<!DOCTYPE html>'):self.url='data:text/html;charset=utf-8,{}'.format(path);self.html=path
		else:print('no support for live URL: %s'%path);sys.exit(1)
	def get_type(self,ele,att='type'):t=ele.attributes.get(att,'');return''if t is None else t
	def remove_elements(self):
		self.tree.body.unwrap_tags(['strong']);tags=['iframe','video','canvas','br'];self.tree.strip_tags(tags)
		for ele in self.tree.css('input[type=file]'):ele.decompose()
		for ele in self.tree.css('div[ondrop]'):ele.decompose()
		for ele in self.tree.css('input[aria-autocomplete]')+[ele for ele in self.tree.css('input')if'autocomplete'in self.get_type(ele,'class')]:ele.decompose()
		scripts=self.tree.css('script');eles=[ele for ele in scripts if'wpf-floatheader.min.js'in ele.html or'form-patch.js'in ele.html or'https://cdn.jotfor.ms/static/prototype.forms.js'in ele.html]
		for ele in eles:ele.decompose()
		for ele in self.tree.css('div.form-captcha'):ele.decompose()
		for ele in self.tree.css('div.formFooter'):ele.decompose()
		for ele in self.tree.css('li')+self.tree.css('span'):
			if len(ele.text())==0:ele.decompose()
		return True
	def get_dv(self,ele):
		try:
			ret_ele=None
			if ele.next is not None and ele.next.tag=='label':ret_ele=ele.next;display_val=ele.next.text()
			elif ele.parent.tag=='label':ret_ele=ele.parent;display_val=ele.parent.text()
			elif ele.next.next is not None and ele.next.next.tag=='label':ret_ele=ele.next.next;display_val=ele.next.next.text()
			else:display_val=''
			return display_val,ret_ele
		except:return None,''
	def fix_radio_listbox(self,elements):
		for ele in elements:
			display_val=None;code_val=None;code_val=ele.attrs['value']if'value'in ele.attrs else''
			if code_val is None or len(code_val)==0:continue
			display_val,display_ele=self.get_dv(ele)
			if display_val is None:self.errored.add(self.path);return False
			if code_val!=display_val:ele.attrs['value']=display_val
		return True
	def add_to_style(self,ele,addition):
		if'style'in ele.attrs:ele.attrs['style']='{}; {}'.format(ele.attrs['style'],addition)
		else:ele.attrs['style']=addition
	def modify_elements(self):
		inputs=[x for x in self.tree.css('input')if self.get_type(x)in['text'or'email']or'type'not in x.attributes]+self.tree.css('textarea');eles=[inp for inp in inputs if'maxlength'not in inp.attributes]
		for ele in eles:ele.attrs['maxlength']=100
		radios=self.tree.css('input[type=radio]');succ=self.fix_radio_listbox(radios)
		if not succ:return False
		checkboxes=self.tree.css('input[type=checkbox]');self.fix_radio_listbox(checkboxes)
		if not succ:return False
		body=self.tree.css_first('body');self.add_to_style(body,'font-family: ocrb; font-size:14px')
		for inp in self.tree.css('textarea'):
			new_inp=HTMLParser('<input type="text">'.format(inp.html)).css_first('input')
			for(k,v)in new_inp.attributes.items():new_inp.attrs[k]=v
			inp.replace_with(new_inp)
		inputs=self.tree.css('input')+self.tree.css('textarea')
		for inp in inputs:
			if'type'not in inp.attrs or'type'in inp.attrs and inp.attrs['type']not in['radio','checkbox']:new_div=HTMLParser('<div style="width: 300px;padding:0px; margin:0px; border:0px;"> {} </div>'.format(inp.html)).css_first('div');inp.replace_with(new_div)
			elif'type'in inp.attrs and inp.attrs['type']in['radio','checkbox']:new_div=HTMLParser('<div style="width: 100%; height:18px;padding:0px; margin:0px; border:0px;"> {} </div>'.format(inp.html)).css_first('div');inp.replace_with(new_div)
		for inp in self.tree.css('input')+self.tree.css('textarea'):
			if'type'in inp.attrs and inp.attrs['type']=='hidden':continue
			if'type'in inp.attrs and inp.attrs['type']in['radio','checkbox']:self.add_to_style(inp,'width:18px; height:18px; font-size:16px; padding:5px; margin:0px; line-height: 16px;')
			else:self.add_to_style(inp,'width: 300px; height:30px; font-size:16px; padding:5px; margin:0px; line-height: 16px;')
		options=self.tree.css('option')
		for ele in options:ele.attrs['value']=ele.text()
		ele=self.tree.css_first('head').child;ele.insert_after(self.minirest);ele.insert_after(self.font_css);ele.insert_after(self.fonts);links=[link for link in self.tree.css('link')if link.html!=self.minirest.html and link.html!=self.font_css.html]
		for link in links:link.decompose()
		for p in self.tree.css('p')+self.tree.css('label'):self.add_to_style(p,'font-family: ocrb; font-size:14px; padding:0px; margin:0px; border:0px;display: block;')
		ps=self.tree.css('button')
		for p in ps:self.add_to_style(p,'padding:20px; vertical-align:bottom; display:inline-block;')
		for p in self.tree.css('h1'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('h2'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('h3'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('h4'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('h5'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('h6'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('span'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('label'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('p'):self.add_to_style(p,'font-size:14px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('div.form-subHeader'):self.add_to_style(p,'font-size:24px; font-family: ocrb;padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('div.form-product-description'):self.add_to_style(p,'padding:5px;')
		for p in self.tree.css('div.widget-inputs-wrapper'):self.add_to_style(p,'height:50px;')
		for p in self.tree.css('select'):self.add_to_style(p,'width:100px;')
		for p in self.tree.css('li')+self.tree.css('tr'):self.add_to_style(p,'padding:5px; margin:0px; border:0px;')
		for p in self.tree.css('td'):self.add_to_style(p,'height:30px;padding:5px; margin:0px; border:0px;')
		return True
	def remove_style(self):
		styles=self.tree.css('style')
		for style in styles:
			sheet=cssutils.parseString(style.html);i=0
			for rule in sheet:
				if rule.type==rule.STYLE_RULE:
					if'tooltip'in rule.selectorText:sheet.deleteRule(i);i+=1;continue
					for pro in rule.style:
						if pro.name.startswith('caret'):rule.style.removeProperty(pro.name)
						elif pro.name.startswith('outline'):rule.style.removeProperty(pro.name)
				i+=1
			new_style='<style>{}</style>'.format(sheet.cssText.decode('utf-8'));style.replace_with(HTMLParser(new_style).css_first('style'))
		return True
	def produce_compatible_page(self):
		if'jfQuestionLabelContainer'in self.html or'<title>This form is currently unavailable!</title>'in self.html:self.errored.add(self.path);return''
		self.html=self.html.replace('width:1px','width:30px');self.html=self.html.replace('width: 1px','width:30px');self.html=self.html.replace('Lucida Grande','Arial');self.html=self.html.replace("content: '';",'');self.tree=HTMLParser(self.html)
		if'Free Online Form Builder & Form Creator | JotForm'in str(self.tree.css_first('title').text()):return''
		if not self.remove_elements():return''
		if not self.modify_elements():return''
		return self.tree.html