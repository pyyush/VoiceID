# VoiceID: An End-to-End Text-Independent Speaker Verification

### Introduction:
It is an end-to-end speaker verification system that aims to confirm the identity of a speaker by matching some representation of an incoming test phrase to that of a set of speaker-dependent enrolment phrases. In this project, we first show an LSTM baseline system proposed in [1]. We then proposed an end-to-end system that works directly on raw waveforms and doesn't require any feature engineering or pre-processing of data.

##### LSTM baseline architecture:
<figure>
  <img src="images/baseline.JPG"/>
</figure>

##### Proposed end-to-end architecture:
<figure>
  <img src="images/E2E.JPG" />
</figure>

### Dataset:
In this project, we are introducing a new limited vocabulary spoken commands dataset which we named 'PS60k'. PS60k has total 60K utterances of 'Hey Siri' and 'Hey Portal' combined. We recorded 60 speakers, 20 from each nationality - China, India and the United States of America. Each speaker speaks 20 different utterences (10 Hey Siri utterances, 10 Hey Portal utterances). These total 1200 original recordings are merged with 10 different types of noises at 5 SNR (-5, 0, 10, 15, 25) levels. PS60k dataset is available on request. To request to the dataset please contact Piyush Vyas at pi.yush@icloud.com or Darshan Shinde at darshinde7802@gmail.com

### Code distribution:
VoiceId repository has three python files. 
1. gen_data.py: This file generate the PS60k dataset from original speaker utterances and 10 noise recordings.
2. baseline.py: This file contains the complete source code for baseline LSTM system, training and testing baseline model.
3. e2e.py: This file contains the complete source code for proposed end-to-end system, training and testing the end-to-end model.

### Basic requirements:
1. To run this model, you will need python 3.7.4 version installed on your system.
2. The uploaded code requires python libraries like librosa, libsndfile, audioread, sklearn, pytorch-1.3.0, cuda-10.1, numpy, etc. installed in your python environment.
3. To train the model, you will need at least 16GB GPU memory. The written code supports multi-GPU training, but all GPUs should be on same node. 

### How to run:
1. Generate data:  
   i.   Clone VoiceID reposity.  
   ii.  Request original dataset from the author.    
   iii. Download and store the dataset inside the cloned repository at the same level where source code is present.  
   iv.  Make a new directory by the name of PS60k.  
   v.   Run gen_data.py to generate the PS60k dataset.   
2. Train and Test baseline LSTM model:  
   i.   If PS60k dataset is ready, then run baseline.py code to train and test the baseline LSTM model.
3. Train and Test propose end-to-end model:  
   i.   If PS60k dataset is ready, then run e2e.py code to train and test the proposed end-to-end model.

### References:
1. E. Marchi, S. Shum, K. Hwang, S. Kajarekar, S. Sigtia, H. Richards, R. Haynes, Y. Kim, and J. Bridle, “Generalised dis-criminative transform via curriculum learning for speaker recognition,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP- 2018)
2. J. W. Jung, H.-S. Heo, J.-h. Kim, H.-J. Shim, and H.-J. Yu, “RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification,” (INTERSPEECH-2019)
