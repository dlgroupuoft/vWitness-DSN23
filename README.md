# vWitness-DSN23
Contains the validation logic, models, training sets of the vWitness paper 

vWitness is a cool idea that uses comptuer vision to "witness" website interactions, allowing a trusted component to assess whether a website, the brower UI or the client system has been tampered with in an interaction between a web server and a user. In essence vWitness acts like a trusted witness, asserting to the website that a) the website was displayed correctly, b) the user interacted with the website in a genuine way and c) the resulting web requests sent to the website match the actions of the user. vWitness is a unique approach for leveraging trusted computing in a way that allows richer UI interactions for the user, while still maintaining the integrity of the website and user interactions with it. 



- server_side: 
  	- segment.py: generats vSPECs
  	- incompatibility.py: addresses incompatible elements
- client_side:
  	- text_validator:
		- data: model training data
	      	- one_font/ocrb: training data for the OCRB model with full and collapsed labels 
				- save.npz:  
				- save_collapsed.npz:
			- full: 
				- save_collapsed.npz
		- code: 
			- train.py: code to reproduce our training and evaluation on the text model 
		- models:
			- ocrb.h5: this is the t6 model as in Table 1 of the paper. 
	    	- full.h5: this is the model trained on all available fonts. 

	- image_validator: 
	  - code
	    - g_baseline_train.py: code to reproduce the model training and robustness evaluation of the image validator
	  - data
	    - cifar_0.npz: training data from cifar 
	    - matrial.npz: training data from material icon, the image verifier is trained with both datasets
	  - model
	    - model_v2.h5: the image model corresponding to g3 of Table 1 of the paper.
	- validation_al:
	  - vWitness.py: performs validation
	  - lib.py:  
