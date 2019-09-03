![travis build](https://api.travis-ci.com/s3nh/pytorch-text-recognition.svg?branch=master)

# Text recognition tool

Text recognition with Pytorch Using CRNN and CRAFT 
pretrained models. 




Main sources:

- [Character Region Awareness for Text Detection Paper](https://arxiv.org/pdf/1904.01941.pdf)
- [CRNN Paper]

Install it, install requirements.txt by 


```python
python -m pip install requirements.txt
```


and check argparsers. at the moment it is only input image so 


```python

> source venv/Scripts/activate
> python app.py --input_file test_image.jpeg

```


### 


Example results 



![image](https://ivrlwww.epfl.ch/research/topics/images/FilteredTextDetection/DollarGlen.jpg)


return .json file with 


``` 
{"0": "dollar", "1": "glen", "2": "and", "3": "campbeli", "4": "castle"}0

``` 

#### Usage

You can clone this repository, 
set up your own virtual environment by 

``` 

python -m venv venv

```

Activate it by 


```
source venv/Scripts/activate
```

Install requirements using ``` pip```


``` 
python -m pip install -r requirements.txt
```

and run ``` app.py ``` 

```

python app.py 

```

then go to 


```localhost:8000``` 

and test it. 


### 
I decided to put it on simplest flask module only 
to show it's basic functionality. 




### To dos


add frontend and some fancy vis. 

