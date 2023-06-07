import cv2,random,numpy as np,sys,os,math
from imutils.object_detection import non_max_suppression
import hashlib,time,datetime,ctypes
from ctypes.util import find_library
import cffi,cProfile,locale
locale.setlocale(locale.LC_ALL,'C')
H_MAX=180
S_MAX=256
V_MAX=256
H_THR=H_MAX*.1
S_THR=S_MAX*.2
V_THR=V_MAX*.2
WHITE_THR=S_THR*.05
MIN_TEXT_SIZE=100
performance_mode=1
api=None
east_p='/home/ss/Documents/opencv-text-recognition/frozen_east_text_detection.pb'
ffi=cffi.FFI()
ffi.cdef('\ntypedef signed char             l_int8;\ntypedef unsigned char           l_uint8;\ntypedef short                   l_int16;\ntypedef unsigned short          l_uint16;\ntypedef int                     l_int32;\ntypedef unsigned int            l_uint32;\ntypedef float                   l_float32;\ntypedef double                  l_float64;\ntypedef long long               l_int64;\ntypedef unsigned long long      l_uint64;\ntypedef int l_ok; /*!< return type 0 if OK, 1 on error */\n\n\nstruct Pix;\ntypedef struct Pix PIX;\ntypedef enum lept_img_format {\n    IFF_UNKNOWN        = 0,\n    IFF_BMP            = 1,\n    IFF_JFIF_JPEG      = 2,\n    IFF_PNG            = 3,\n    IFF_TIFF           = 4,\n    IFF_TIFF_PACKBITS  = 5,\n    IFF_TIFF_RLE       = 6,\n    IFF_TIFF_G3        = 7,\n    IFF_TIFF_G4        = 8,\n    IFF_TIFF_LZW       = 9,\n    IFF_TIFF_ZIP       = 10,\n    IFF_PNM            = 11,\n    IFF_PS             = 12,\n    IFF_GIF            = 13,\n    IFF_JP2            = 14,\n    IFF_WEBP           = 15,\n    IFF_LPDF           = 16,\n    IFF_TIFF_JPEG      = 17,\n    IFF_DEFAULT        = 18,\n    IFF_SPIX           = 19\n};\n\nchar * getLeptonicaVersion (  );\nPIX * pixRead ( const char *filename );\nl_int32 pixGetDimensions ( PIX *pix, l_int32 *pw, l_int32 *ph, l_int32 *pd );\nPIX * pixCreate ( int width, int height, int depth );\nPIX * pixEndianByteSwapNew(PIX  *pixs);\nl_int32 pixSetData ( PIX *pix, l_uint32 *data );\nl_ok pixSetPixel ( PIX *pix, l_int32 x, l_int32 y, l_uint32 val );\nl_ok pixWrite ( const char *fname, PIX *pix, l_int32 format );\nl_int32 pixFindSkew ( PIX *pixs, l_float32 *pangle, l_float32 *pconf );\nPIX * pixDeskew ( PIX *pixs, l_int32 redsearch );\nvoid pixDestroy ( PIX **ppix );\nl_ok pixGetResolution ( const PIX *pix, l_int32 *pxres, l_int32 *pyres );\nl_ok pixSetResolution ( PIX *pix, l_int32 xres, l_int32 yres );\nl_int32 pixGetWidth ( const PIX *pix );\n\ntypedef struct TessBaseAPI TessBaseAPI;\ntypedef struct ETEXT_DESC ETEXT_DESC;\ntypedef struct TessPageIterator TessPageIterator;\ntypedef struct TessResultIterator TessResultIterator;\ntypedef int BOOL;\n\ntypedef enum TessOcrEngineMode  {\n    OEM_TESSERACT_ONLY          = 0,\n    OEM_LSTM_ONLY               = 1,\n    OEM_TESSERACT_LSTM_COMBINED = 2,\n    OEM_DEFAULT                 = 3} TessOcrEngineMode;\n\ntypedef enum TessPageSegMode {\n    PSM_OSD_ONLY               =  0,\n    PSM_AUTO_OSD               =  1,\n    PSM_AUTO_ONLY              =  2,\n    PSM_AUTO                   =  3,\n    PSM_SINGLE_COLUMN          =  4,\n    PSM_SINGLE_BLOCK_VERT_TEXT =  5,\n    PSM_SINGLE_BLOCK           =  6,\n    PSM_SINGLE_LINE            =  7,\n    PSM_SINGLE_WORD            =  8,\n    PSM_CIRCLE_WORD            =  9,\n    PSM_SINGLE_CHAR            = 10,\n    PSM_SPARSE_TEXT            = 11,\n    PSM_SPARSE_TEXT_OSD        = 12,\n    PSM_COUNT                  = 13} TessPageSegMode;\n\ntypedef enum TessPageIteratorLevel {\n     RIL_BLOCK    = 0,\n     RIL_PARA     = 1,\n     RIL_TEXTLINE = 2,\n     RIL_WORD     = 3,\n     RIL_SYMBOL    = 4} TessPageIteratorLevel;    \n\nTessPageIterator* TessBaseAPIAnalyseLayout(TessBaseAPI* handle);\nTessPageIterator* TessResultIteratorGetPageIterator(TessResultIterator* handle);\n\nBOOL TessPageIteratorNext(TessPageIterator* handle, TessPageIteratorLevel level);\nBOOL TessPageIteratorBoundingBox(const TessPageIterator* handle, TessPageIteratorLevel level,\n                                 int* left, int* top, int* right, int* bottom);\n\nconst char* TessVersion();\nvoid TessBaseAPISetOutputName(TessBaseAPI* handle, const char* name);\nint* TessBaseAPIAllWordConfidences(TessBaseAPI* handle);\n\nTessBaseAPI* TessBaseAPICreate();\nint    TessBaseAPIInit3(TessBaseAPI* handle, const char* datapath, const char* language);\nint    TessBaseAPIInit2(TessBaseAPI* handle, const char* datapath, const char* language, TessOcrEngineMode oem);\nvoid   TessBaseAPISetPageSegMode(TessBaseAPI* handle, TessPageSegMode mode);\nvoid   TessBaseAPISetImage(TessBaseAPI* handle,\n                           const unsigned char* imagedata, int width, int height,\n                           int bytes_per_pixel, int bytes_per_line);\nvoid   TessBaseAPISetImage2(TessBaseAPI* handle, struct Pix* pix);\n\nBOOL   TessBaseAPISetVariable(TessBaseAPI* handle, const char* name, const char* value);\nBOOL   TessBaseAPIDetectOrientationScript(TessBaseAPI* handle, char** best_script_name, \n                                                            int* best_orientation_deg, float* script_confidence, \n                                                            float* orientation_confidence);\nint TessBaseAPIRecognize(TessBaseAPI* handle, ETEXT_DESC* monitor);\nTessResultIterator* TessBaseAPIGetIterator(TessBaseAPI* handle);\nBOOL   TessResultIteratorNext(TessResultIterator* handle, TessPageIteratorLevel level);\nchar*  TessResultIteratorGetUTF8Text(const TessResultIterator* handle, TessPageIteratorLevel level);\nfloat  TessResultIteratorConfidence(const TessResultIterator* handle, TessPageIteratorLevel level);\nchar*  TessBaseAPIGetUTF8Text(TessBaseAPI* handle);\nconst char*  TessResultIteratorWordFontAttributes(const TessResultIterator* handle, BOOL* is_bold, BOOL* is_italic,\n                                                              BOOL* is_underlined, BOOL* is_monospace, BOOL* is_serif,\n                                                              BOOL* is_smallcaps, int* pointsize, int* font_id);\nvoid   TessBaseAPIEnd(TessBaseAPI* handle);\nvoid   TessBaseAPIDelete(TessBaseAPI* handle);\n')
tess_libname='/usr/local/lib/libtesseract.so.4'
lept_libname='/usr/lib/x86_64-linux-gnu/liblept.so.5'
tesseract=ffi.dlopen(tess_libname)
leptonica=ffi.dlopen(lept_libname)
ffi_ocrad=cffi.FFI()
ffi_ocrad.cdef('\n\n\n/* OCRAD_Pixmap.data is a pointer to image data formed by "height" rows\n   of "width" pixels each.\n   The format for each pixel depends on mode like this:\n   OCRAD_bitmap   --> 1 byte  per pixel;  0 = white, 1 = black\n   OCRAD_greymap  --> 1 byte  per pixel;  256 level greymap (0 = black)\n   OCRAD_colormap --> 3 bytes per pixel;  16777216 colors RGB (0,0,0 = black) */\n\nenum OCRAD_Pixmap_Mode { OCRAD_bitmap, OCRAD_greymap, OCRAD_colormap };\n\nstruct OCRAD_Pixmap\n  {\n  const unsigned char * data;\n  int height;\n  int width;\n  enum OCRAD_Pixmap_Mode mode;\n  };\n\n\nenum OCRAD_Errno { OCRAD_ok = 0, OCRAD_bad_argument, OCRAD_mem_error,\n                   OCRAD_sequence_error, OCRAD_library_error };\n\nstruct OCRAD_Descriptor;\n\n\nconst char * OCRAD_version( void );\n\n\n/*--------------------- Functions ---------------------*/\n\nstruct OCRAD_Descriptor * OCRAD_open( void );\n\nint OCRAD_close( struct OCRAD_Descriptor * const ocrdes );\n\nenum OCRAD_Errno OCRAD_get_errno( struct OCRAD_Descriptor * const ocrdes );\n\nint OCRAD_set_image( struct OCRAD_Descriptor * const ocrdes,\n                     const struct OCRAD_Pixmap * const image,\n                     const bool invert );\n\nint OCRAD_set_image_from_file( struct OCRAD_Descriptor * const ocrdes,\n                               const char * const filename,\n                               const bool invert );\n\nint OCRAD_set_utf8_format( struct OCRAD_Descriptor * const ocrdes,\n                           const bool utf8 );\t\t// 0 = byte, 1 = utf8\n\nint OCRAD_set_threshold( struct OCRAD_Descriptor * const ocrdes,\n                         const int threshold );\t\t// 0..255, -1 = auto\n\nint OCRAD_scale( struct OCRAD_Descriptor * const ocrdes, const int value );\n\nint OCRAD_recognize( struct OCRAD_Descriptor * const ocrdes,\n                     const bool layout );\n\nint OCRAD_result_blocks( struct OCRAD_Descriptor * const ocrdes );\n\nint OCRAD_result_lines( struct OCRAD_Descriptor * const ocrdes,\n                        const int blocknum );\t\t// 0..blocks-1\n\nint OCRAD_result_chars_total( struct OCRAD_Descriptor * const ocrdes );\n\nint OCRAD_result_chars_block( struct OCRAD_Descriptor * const ocrdes,\n                              const int blocknum );\t// 0..blocks-1\n\nint OCRAD_result_chars_line( struct OCRAD_Descriptor * const ocrdes,\n                             const int blocknum,\t// 0..blocks-1\n                             const int linenum );\t// 0..lines(block)-1\n\nconst char * OCRAD_result_line( struct OCRAD_Descriptor * const ocrdes,\n                                const int blocknum,\t// 0..blocks-1\n                                const int linenum );\t// 0..lines(block)-1\n\nint OCRAD_result_first_character( struct OCRAD_Descriptor * const ocrdes );\n')
ocrad_libname='/home/ss/Documents/ocrad-0.27/libocrad.so'
ocrad=ffi_ocrad.dlopen(ocrad_libname)
ocrdes=None
show_debug_imgs=1
if len(sys.argv)>3:show_debug_imgs=int(sys.argv[3])
use_cache=0
if use_cache:text_cache=Cache();color_cache=Cache();img_cache=Cache()
cached_exp=None
cached_loc=None
nets=None
pr=cProfile.Profile()
def get_abs_path_of_library(library):
	'Get absolute path of library.';abs_path=None;lib_name=find_library(library)
	if os.path.exists(lib_name):abs_path=os.path.abspath(lib_name);return abs_path
	libdl=ctypes.CDLL(lib_name)
	if not libdl:return abs_path
	try:dlinfo=libdl.dlinfos
	except AttributeError as err:abs_path=str(err).split(':')[0]
	return abs_path
def bgr2PIX32(gray,leptonica):'Convert opencv bgr to PIX';h,w=gray.shape;img=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGBA);data=img.tobytes();pixs=leptonica.pixCreate(w,h,32);leptonica.pixSetData(pixs,ffi.from_buffer('l_uint32[]',data));return leptonica.pixEndianByteSwapNew(pixs)
def gray2pgm(gray):assert len(gray.shape)==2;h,w=gray.shape;pixmap=ffi_ocrad.new('struct OCRAD_Pixmap *');pixmap.mode=ocrad.OCRAD_greymap;pixmap.height=h;pixmap.width=w;flat=list(gray.ravel());pixmap.data=ffi_ocrad.new('unsigned char []',flat);return pixmap
def actual_ocr_ocrad(gray):
	global ocrdes
	if ocrdes is None:ocrdes=ocrad.OCRAD_open();ocrad.OCRAD_set_threshold(ocrdes,-1)
	pixmap=gray2pgm(gray);ocrad.OCRAD_set_image(ocrdes,pixmap,False);ocrad.OCRAD_recognize(ocrdes,False);ret=b'';blocks=int(ocrad.OCRAD_result_blocks(ocrdes))
	for b in range(blocks):
		lines=int(ocrad.OCRAD_result_lines(ocrdes,b))
		for l in range(lines):
			s=ffi_ocrad.string(ocrad.OCRAD_result_line(ocrdes,b,l))
			if s:ret+=s
		if b+1<blocks:ret+='\n'
	return ret
def remove_api():tesseract.TessBaseAPIEnd(api)
def actual_ocr_v2(gray):
	pix=bgr2PIX32(gray,leptonica);global api
	if api is None:api=tesseract.TessBaseAPICreate();tesseract.TessBaseAPIInit2(api,'/usr/local/share/tessdata/'.encode(),'eng'.encode(),tesseract.OEM_TESSERACT_LSTM_COMBINED);tesseract.TessBaseAPISetPageSegMode(api,tesseract.PSM_AUTO)
	tesseract.TessBaseAPISetImage2(api,pix);tesseract.TessBaseAPIRecognize(api,ffi.NULL);ret=tesseract.TessBaseAPIAllWordConfidences(api);utf8_text='';recgonized_val=tesseract.TessBaseAPIGetUTF8Text(api)
	if recgonized_val!=ffi.NULL:utf8_text=ffi.string(recgonized_val).decode('utf-8')
	conf=0
	if ret==ffi.NULL or ret[0]<40:utf8_text='';conf=ret[0]
	del pix;return utf8_text,conf
class Cache:
	def __init__(self):self.cache={};self.max_cache_size=1000;self.cache_hit=0
	def hit_img(self,img):key=hash(str(img));return key in self.cache
	def update_img(self,img,value):
		key=hash(str(img))
		if key not in self.cache and len(self.cache)>=self.max_cache_size:self.remove_oldest()
		self.cache[key]={'last_accessed':datetime.datetime.now(),'val':value}
	def get_img(self,img):key=hash(str(img));self.cache[key]['last_accessed']=datetime.datetime.now();self.cache_hit+=1;return self.cache[key]['val']
	def __contains__(self,key):'\n        Returns True or False depending on whether or not the key is in the \n        cache\n        ';return key in self.cache
	def __getitem__(self,key):self.cache[key]['last_accessed']=datetime.datetime.now();self.cache_hit+=1;return self.cache[key]['val']
	def update(self,key,value):
		'\n        Update the cache dictionary and optionally remove the oldest item\n        '
		if key not in self.cache and len(self.cache)>=self.max_cache_size:self.remove_oldest()
		self.cache[key]={'last_accessed':datetime.datetime.now(),'val':value}
	def remove_oldest(self):
		'\n        Remove the entry that has the oldest accessed date\n        ';oldest_entry=None
		for key in self.cache:
			if oldest_entry is None:oldest_entry=key
			elif self.cache[key]['last_accessed']<self.cache[oldest_entry]['last_accessed']:oldest_entry=key
		self.cache.pop(oldest_entry)
	@property
	def size(self):'\n        Return the size of the cache\n        ';return len(self.cache)
MAX_FEATURES=500
GOOD_MATCH_PERCENT=.15
def find_pts(path):img=cv2.imread(path);gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);gray=np.float32(gray);dst=cv2.cornerHarris(gray,2,3,.04);dst=cv2.dilate(dst,None);ret,dst=cv2.threshold(dst,.01*dst.max(),255,0);dst=np.uint8(dst);ret,labels,stats,centroids=cv2.connectedComponentsWithStats(dst);criteria=cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,.001;corners=cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria);res=np.hstack((centroids,corners));res=np.int0(res);img[(res[:,1],res[:,0])]=[0,0,255];img[(res[:,3],res[:,2])]=[0,255,0];return img
def find_large_bb(im):
	img_h,img_w,_=im.shape;imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);avg_grayscale=math.floor(np.average(imgray))
	if avg_grayscale<100:ret,thresh=cv2.threshold(imgray,avg_grayscale,255,cv2.THRESH_BINARY)
	else:ret,thresh=cv2.threshold(imgray,avg_grayscale,255,cv2.THRESH_BINARY_INV)
	dilated=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)));contours,_=cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);new_contours=[];size_thr=img_w*img_h*.9
	for c in contours:
		if cv2.contourArea(c)<size_thr:new_contours.append(c)
	best_box=[-1,-1,-1,-1];mask=np.zeros((img_h,img_w),np.uint8)
	for c in new_contours:
		x,y,w,h=cv2.boundingRect(c);x,y,x2,y2=max(x-2,0),max(y-2,0),min(x+w+2,img_w),min(y+h+2,img_h);cv2.rectangle(mask,(x,y),(x2,y2),255,-1)
		if best_box[0]<0:best_box=[x,y,x2,y2]
		else:
			if x<best_box[0]:best_box[0]=x
			if y<best_box[1]:best_box[1]=y
			if x2>best_box[2]:best_box[2]=x2
			if y2>best_box[3]:best_box[3]=y2
	if sum(best_box)==-4:return im,-1,None
	x1,y1,x2,y2=best_box;horizontal_size=img_w//30;mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(horizontal_size,1)));contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);res=[]
	for c in contours:x,y,w,h=cv2.boundingRect(c);cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1)
	return im[y1:y2,x1:x2],(x1,x2,y1,y2),mask[y1:y2,x1:x2]
def resize(img1,img2):
	method=min;method=max;h1,w1,_=img1.shape;h2,w2,_=img2.shape
	if h1==h2 and w1==w2:return img1,img2
	elif w1<w2 and h1<h2 or w2<w1 and h2<h1:
		target_dim=method(w1,w2),method(h1,h2);to_scale,no_scale=img1,img2
		if(img1.shape[1],img1.shape[0])==target_dim:to_scale,no_scale=img2,img1
	else:target_dim=w2,h2;to_scale,no_scale=img1,img2
	if method==max:scaled=cv2.resize(to_scale,target_dim,interpolation=cv2.INTER_CUBIC)
	else:scaled=cv2.resize(to_scale,target_dim,interpolation=cv2.INTER_AREA)
	return scaled,no_scale
def decode_predictions(scores,geometry):
	min_confidence=.5;numRows,numCols=scores.shape[2:4];rects=[];confidences=[]
	for y in range(0,numRows):
		scoresData=scores[(0,0,y)];xData0=geometry[(0,0,y)];xData1=geometry[(0,1,y)];xData2=geometry[(0,2,y)];xData3=geometry[(0,3,y)];anglesData=geometry[(0,4,y)]
		for x in range(0,numCols):
			if scoresData[x]<min_confidence:continue
			offsetX,offsetY=x*4.,y*4.;angle=anglesData[x];cos=np.cos(angle);sin=np.sin(angle);h=xData0[x]+xData2[x];w=xData1[x]+xData3[x];endX=int(offsetX+cos*xData1[x]+sin*xData2[x]);endY=int(offsetY-sin*xData1[x]+cos*xData2[x]);startX=int(endX-w);startY=int(endY-h);rects.append((startX,startY,endX,endY));confidences.append(scoresData[x])
	return rects,confidences
def text_detect_east(image):
	padding=0;orig=image.copy();origH,origW=image.shape[:2];newH=max(origH-origH%32,32);newW=max(origW-origW%32,32);rW=origW/float(newW);rH=origH/float(newH);image=cv2.resize(image,(newW,newH));layerNames=['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3'];global nets
	if nets is not None and(newW,newH)in nets:net=nets[(newW,newH)]
	else:
		net=cv2.dnn.readNet(east_p)
		if nets is None:nets={}
		nets[(newW,newH)]=net
	blob=cv2.dnn.blobFromImage(image,1.,(newW,newH),(123.68,116.78,103.94),swapRB=True,crop=False);net.setInput(blob);scores,geometry=net.forward(layerNames);rects,confidences=decode_predictions(scores,geometry);boxes=non_max_suppression(np.array(rects),probs=confidences);results=[]
	for(startX,startY,endX,endY)in boxes:startX=int(startX*rW);startY=int(startY*rH);endX=int(endX*rW);endY=int(endY*rH);dX=int((endX-startX)*padding);dY=int((endY-startY)*padding);startX=max(0,startX-dX);startY=max(0,startY-dY);endX=min(origW,endX+dX*2);endY=min(origH,endY+dY*2);results.append((startX,startY,endX,endY+2))
	return results
def add_boarder(img):top=int(.5*img.shape[0]);bottom=top;left=int(.3*img.shape[1]);right=left;return cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=255)
def enlarge(img):
	target_h=80;h,w=img.shape;ratio=target_h/h;target_w=int(w*ratio)
	if h<target_h:img=cv2.resize(img,(target_w,target_h),interpolation=cv2.INTER_CUBIC)
	return img
def is_colored_background(img,threshold=140):
	'\n    return true if is has a colored (non-white) background\n    ';hist=cv2.calcHist([img],[0],None,[256],[0,256]);total_pix=img.shape[0]*img.shape[1];num_white=sum(hist[250:])[0]
	if num_white<total_pix*.2 and num_white>total_pix*.01:threshold=180
	return math.floor(np.average(img))<=threshold
def get_white_bg(img):
	if is_colored_background(img):return cv2.bitwise_not(img)
	return img
def enlarge_text(img,should_add_boarder=1):
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);img=enlarge(img);img=get_white_bg(img)
	if should_add_boarder:img=add_boarder(img)
	return img
def ocr_run(text_area):
	h,w,_=text_area.shape
	if h*w<MIN_TEXT_SIZE:return'',0
	if use_cache:
		ckey=hashlib.md5(text_area).hexdigest()
		if ckey in text_cache:return text_cache[ckey]
	text_area=enlarge_text(text_area);_,text_area=cv2.threshold(text_area,170,255,cv2.THRESH_BINARY);text,conf=actual_ocr_v2(text_area);text=text.encode('ascii','ignore').decode()
	if use_cache:text_cache.update(ckey,(text,conf))
	return text,conf
def show(titles,imgs,stop=0,wait=0):
	if not show_debug_imgs:return
	for i in range(len(imgs)):
		title=str(i)
		if titles:title=titles[i]
		cv2.imwrite('tmp/{0}.png'.format(title),imgs[i])
	if wait:cv2.waitKey(0)
	if stop:sys.exit(0)
def merge_overlapping_rects(shape,rects):
	h,w,_=shape;tmp=np.zeros((h,w),np.uint8);hori_padding=-.05;verti_padding=.05
	for(x1,y1,x2,y2)in rects:cv2.rectangle(tmp,(x1,y1),(x2,y2),255,-1)
	horizontal_size=max(w//40,1);tmp=cv2.dilate(tmp,cv2.getStructuringElement(cv2.MORPH_RECT,(horizontal_size,1)));contours,_=cv2.findContours(tmp,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);res=[]
	for c in contours:
		x,y,w,h=cv2.boundingRect(c);padding_x=int((x2-x1)*hori_padding);padding_y=int((y2-y1)*verti_padding)
		if cv2.contourArea(c)>MIN_TEXT_SIZE:res.append((x-padding_x,y-padding_y,x+w+2*padding_x,y+h+2*padding_y))
	return res
def text_detect_run(img):
	h,w,_=img.shape
	if h*w<MIN_TEXT_SIZE:return[]
	rects=text_detect_east(img);rects=merge_overlapping_rects(img.shape,rects);return rects;td=lib.TextDetection(local_f)
def to_odd(num):
	if num%2==1:return num
	return max(num-1,0)
def remove_h_v_lines(img):
	'\n    TODO\n    ';img=img.astype(np.uint8);ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY);h,w=img.shape;horizontal_size=to_odd(w//100)
	if horizontal_size>0:horizontalStructure=cv2.getStructuringElement(cv2.MORPH_RECT,(horizontal_size,3));img=cv2.morphologyEx(img,cv2.MORPH_OPEN,horizontalStructure)
	verticalsize=to_odd(h//100)
	if verticalsize>0:verticalStructure=cv2.getStructuringElement(cv2.MORPH_RECT,(3,verticalsize));img=cv2.morphologyEx(img,cv2.MORPH_OPEN,verticalStructure)
	return img
def retrive_edges(src):gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY);return cv2.Canny(gray,100,200)
def color_comparison(img1,img2):
	'\n    img1 is loc\n    img2 is exp\n    '
	if use_cache:
		ckey=hashlib.md5(np.concatenate((img1,img2),axis=0)).hexdigest()
		if ckey in color_cache:return color_cache[ckey]
	show(['img1','img2'],[img1,img2]);rbg_img1=img1.copy();rbg_img2=img2.copy();img1=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV);img2=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV);assert len(img1.shape)>2 and img1.shape[2]==3 and len(img2.shape)>2 and img2.shape[2]==3 and img1.shape[:2]==img2.shape[:2],'color_comparison shape, color mismatch';hsv_diff=cv2.absdiff(img1,img2);bgr_diff=np.sum(cv2.absdiff(rbg_img1,rbg_img2),axis=2);h_d,s_d,v_d=cv2.split(hsv_diff);h_res=np.where(h_d>H_MAX/2,H_MAX-h_d>H_THR,h_d>H_THR);s_res=s_d>S_THR;v_res=v_d>V_THR;h_res=h_res.astype(np.uint8);s_res=s_res.astype(np.uint8);v_res=v_res.astype(np.uint8);h_res=remove_h_v_lines(h_res);h_res=np.where(h_res,bgr_diff>10,0);h_res=np.where(h_res,255,0);h_res=h_res.astype(np.uint8);s_res=remove_h_v_lines(s_res);s_res=np.where(s_res,bgr_diff>15,0);s_res=np.where(s_res,255,0);s_res=s_res.astype(np.uint8);v_res=remove_h_v_lines(v_res);bgr_diff=np.where(bgr_diff,255,0);show(['rbg_img1','rbg_img2'],[rbg_img1,rbg_img2],wait=0);show(['h_diff','s_diff','v_diff'],[h_d,s_d,v_d]);show(['h_diff3','s_diff3','v_diff3','bgr_diff'],[h_res,s_res,v_res,bgr_diff],wait=0);ret=np.count_nonzero(h_res),np.count_nonzero(s_res),np.count_nonzero(v_res),np.count_nonzero(np.add(h_res,s_res,v_res))
	if use_cache:color_cache.update(ckey,ret)
	return ret
def preprocess(s):s=s.lower();s=s.replace('\n','');s=s.replace(' ','');s=s.replace('|','1');s=s.replace('i','1');s=s.replace('I','1');s=s.replace('o','0');s=s.replace('O','0');s=s.replace('s','5');s=s.replace('S','5');return''.join(x for x in s if x.isalnum())
def levenshtein(seq1,seq2):
	size_x=len(seq1)+1;size_y=len(seq2)+1;matrix=np.zeros((size_x,size_y))
	for x in range(size_x):matrix[(x,0)]=x
	for y in range(size_y):matrix[(0,y)]=y
	for x in range(1,size_x):
		for y in range(1,size_y):
			if seq1[x-1]==seq2[y-1]:matrix[(x,y)]=min(matrix[(x-1,y)]+1,matrix[(x-1,y-1)],matrix[(x,y-1)]+1)
			else:matrix[(x,y)]=min(matrix[(x-1,y)]+1,matrix[(x-1,y-1)]+1,matrix[(x,y-1)]+1)
	return matrix[(size_x-1,size_y-1)]
def text_comparison(text1,text2):t1=preprocess(text1);t2=preprocess(text2);l_dis=levenshtein(t1,t2);max_allow=min(max(len(t1),len(t2))/5,3);res=l_dis<=max_allow;return res
def remove_h_v_lines_text(img):img=img.astype(np.uint8);ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY);h,w=img.shape;print(img.shape);kernel=np.ones((3,3),np.uint8);img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel);return img
def pixel_compare_text(im1,im2):
	im1=enlarge_text(im1,0);im2=enlarge_text(im2,0);hist1=cv2.calcHist([im1],[0],None,[256],[0,256]);cv2.normalize(hist1,hist1,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);hist2=cv2.calcHist([im2],[0],None,[256],[0,256]);cv2.normalize(hist2,hist2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);res=cv2.compareHist(hist1,hist2,0);threshold=.997
	if show_debug_imgs and res<threshold:print(res)
	return res>threshold
def pixel_compare_text2(im1,im2):
	im1=cv2.cvtColor(im1,cv2.COLOR_BGR2HSV);im2=cv2.cvtColor(im2,cv2.COLOR_BGR2HSV);h1,s1,v1=cv2.split(im1);h2,s2,v2=cv2.split(im2);hist_h=cv2.calcHist([im1],[0],None,[180],[0,180]);cv2.normalize(hist_h,hist_h,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);hist_s=cv2.calcHist([im1],[1],None,[256],[0,256]);cv2.normalize(hist_s,hist_s,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);hist_v=cv2.calcHist([im1],[2],None,[256],[0,256]);cv2.normalize(hist_v,hist_v,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);hist_h2=cv2.calcHist([im2],[0],None,[180],[0,180]);cv2.normalize(hist_h2,hist_h2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);hist_s2=cv2.calcHist([im2],[1],None,[256],[0,256]);cv2.normalize(hist_s2,hist_s2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);hist_v2=cv2.calcHist([im2],[2],None,[256],[0,256]);cv2.normalize(hist_v2,hist_v2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);comp_method=0;res=min(cv2.compareHist(hist_h,hist_h2,comp_method),cv2.compareHist(hist_s,hist_s2,comp_method),cv2.compareHist(hist_v,hist_v2,comp_method));rand=random.randint(1,100);threshold=.997
	if show_debug_imgs and res<threshold:print(rand,res);show(['{}_img1'.format(rand)],[im1]);show(['{}_img2'.format(rand)],[im2])
	return res>threshold;channels=[0];hist=cv2.calcHist([im1],channels,None,[256],[0,256]);cv2.normalize(hist,hist,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);hist2=cv2.calcHist([im2],channels,None,[256],[0,256]);cv2.normalize(hist2,hist2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX);res=cv2.compareHist(hist,hist2,0);rand=random.randint(1,100);threshold=.997
	if show_debug_imgs and res<threshold:print(rand,res);show(['{}_img1'.format(rand)],[im1]);show(['{}_img2'.format(rand)],[im2])
	return res>threshold;hsv_diff=cv2.absdiff(cv2.cvtColor(im1,cv2.COLOR_BGR2HSV),cv2.cvtColor(im2,cv2.COLOR_BGR2HSV));h_d,s_d,v_d=cv2.split(hsv_diff);h_res=np.where(h_d>H_MAX/2,H_MAX-h_d>H_THR,h_d>H_THR);s_res=s_d>S_THR;v_res=v_d>V_THR;bgr_diff=np.sum(cv2.absdiff(img1,img2),axis=2);h_res=np.where(h_res,bgr_diff>10,0);h_res=np.where(h_res,255,0);h_res=h_res.astype(np.uint8);h_res=remove_h_v_lines_text(h_res);s_res=remove_h_v_lines_text(s_res);v_res=remove_h_v_lines_text(v_res);res=h_res+s_res+v_res;show(['res{}'.format(time.time())],[res]);show(['img1{}'.format(time.time())],[img1]);show(['img2{}'.format(time.time())],[img2]);non_zero=np.count_nonzero(res)
	if non_zero>0 and show_debug_imgs:print(np.count_nonzero(h_res),np.count_nonzero(s_res),np.count_nonzero(v_res),np.count_nonzero(res),h_res.shape[0]*h_res.shape[1])
	return non_zero==0