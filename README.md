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

and wil save it in default path. 

### Installing 


python setup.py install 

## Deployment

to do : 

add docker/flask/redis to store and visualize results

### To dos


Flask app
Bootstrap frontend 

docker compose 


onxx it and put on mobile.
how to make it faster? (todo)


## Versioning
0.1a

## Authors


## Acknowledments 

Input image (TODO: Preprocess origin format)
Output - file with predicted words  on input file


```python
Image -> CRAFT -> CRNN -> .json

```
