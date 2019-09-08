Deployed in here, 

[Link](https://glacial-ravine-89423.herokuapp.com/)

but It will exceed an memory error on prediciton, so I'll push it to ews instance. 

![travis build](https://api.travis-ci.com/s3nh/pytorch-text-recognition.svg?branch=master)

### Text detection and recognition
This repository contains tool which allow to detect region with text and translate it one by one. 


### Description
Two pretrained neural networks are used. One of them is responsible for detecting places in which 
text appear and return its coordinates. 
Structure use for this operation is based on CRAFT architecture. 
- [Craft Paper](https://arxiv.org/pdf/1904.01941.pdf)

Second network take detected words and recognize words included inside it. 
Convolutional Recurrential neural networks  (CRNN) are used for this operation. 

- [CRNN Paper](https://arxiv.org/abs/1507.05717)


#### Deployment 
I decided to deploy it on heroku (temporarily solution), but the amount of memory available on this platform
is not enough. 
You can check it on [heroku app](https://glacial-ravine-89423.herokuapp.com/).
I decided to add bootstrap template because whole solution become more intuitive.

### Windows Installation
To install it locally, you can run from your virtual env

```python
python -m pip install requirements.txt
```

#### Linux installation

to install it properly on Linux OS you have to install additionaly 


```buildoutcfg

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

```

If problems with cv2 imports are still appearing then you should install 


```buildoutcfg
pip install opencv-contrib-python
```

Then you can run 

```buildoutcfg
```python
python -m pip install requirements.txt
```

### Run 
To run it locally, please activate your environment 

```buildoutcfg
> win
venv\Scripts\activate.bat

>linux
source venv\Scripts\activate

```
and run straight from project origin


```buildoutcfg
python  app.py

```
If everything goes properly, you'll see on localhost:8000, 
screen just like one below.

![screen](img/front_.PNG?raw=True)




#### Updates

I decided to remove argparse, because as I mention earlier, it was less intuitive. 
Solution is not fast, is more like an toy example which shows how to use Pytorch model 
on deployment environment. 

Version which I use here contain torch-cpu which make preprocessing and detecting slightly slower. 
I test it on cuda and it was much faster.

If you have more information, drop me a line
If you like it, give a star 

[Contact Info](https://s3nh.github.io)





