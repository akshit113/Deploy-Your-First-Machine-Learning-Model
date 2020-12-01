# Workflow for exposing an ML Project into a REST API using FastAPI framework and Heroku
In this project, you will learn step-by-step about how to spin up an API for a ML project using FastAPI framework and Heroku PaaS. 
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python.  
Heroku is a container-based cloud Platform as a Service (PaaS). Developers use Heroku to deploy, manage, and scale modern apps.

In this example, we have used [Iris Dataset](http://archive.ics.uci.edu/ml/datasets/Iris/), which is a very popular ML toy problem.
The goal here is to understand to the end-to-end deployment process of a Machine Learning project. The final web application is hosted at this [Heroku](https://predict-iris-flower-species.herokuapp.com/docs) link

## Description of the Dataset
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. 
Predicted attribute: class of iris plant. In this project, the dataset is sourced from scikit-learn library using `from sklearn.datasets import load_iris`. This dataset can also be found at UCI Machine Repository website here: http://archive.ics.uci.edu/ml/datasets/Iris/

## Features
**Input**: sepal length in cm, sepal width in cm, petal length in cm, petal width in cm. <br/>
**Output**: Multiclass classification (n=3), Iris Setosa, Iris Versicolour, Iris Virginica 

## Dependencies
python, numpy, pandas, sckit-learn, fastapi, pydantic, uvicorn, pickle

## Project Structure
predict_iris_flower_species/<br/>
    |____ .git<br/>
    |____ venv/<br/>
    |____ __init__.py <br/>
    |____ README.md <br/>
    |____ Procfile <br/>
    |____ .gitignore <br/>
    |____ requirements.txt <br/>
    |____ model.py <br/>
    |____ app.py <br/>
    |____ iris.py <br/>
    |____ classifier.pkl <br/>

## Steps for Deployment

1. Create model.py
    - Create model.py to load and clean dataset
    - build and evaluate your model to get decent accuracy
    - Export the model into a pickle (*.pkl) file

2. Create iris.py to create a class which holds the HTTP Form POST parameters
    - Use `pydantic` module to create a template of input features to the model
    - pydantic enforces type hints at runtime, and provides user friendly errors when data is invalid.

3. Create app.py
    - import packages
        ```python
        import pickle
        from iris import Iris
        import uvicorn
        from fastapi import FastAPI
        ```
    - Instantiate the FastAPI class as an 'app' object
       ```python
       app = FastAPI()
       pickle_in = open('classifier.pkl', 'rb')
       classifier = pickle.load(pickle_in)
       ```
    - create index and predict endpoints with GET and POST request methods
        ```python
        @app.get('/')
        def index():
            return {'message': 'hello, stranger'}

        @app.post('/predict')
        def predict_flower(data: Iris):
            pass
        ```
    - In predict method, 
        - pass the function parameter data:Iris which is basically our HTTP POST parameters from the form
        - Create the convert the data into dictionary using data.dict()
        - parse the dictionary to get the input values and create an 1D numpy array
            ```python
            data = data.dict()
            sepal_length = data['sepal_width']
            sepal_width = data['sepal_length']
            petal_length = data['petal_length']
            petal_width = data['petal_width']
            ```
        - finally pass the 1-D numpy array to our model.predict() function
            ```python
            prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
            ```
        - use if-else logic to convert the output into useful information in the prediction variable
            ```python
            if (prediction[0] == 0):
                prediction = "Iris Veriscolor (0)"
            elif (prediction[0] == 1):
                prediction = "Iris Setosa (1)"
            elif (prediction[0] == 2):
                prediction = 'Iris Virginica (2)'
            else:
                prediction = 'Not a valid prediction, check input'
            ```
        - return the prediction in json notation. For example,
            ```python
            return {
            'prediction': prediction
            }
            ```
            
        - App driver code
            ```python
            if __name__ == '__main__':
                uvicorn.run(app, host='127.0.0.1', port=8000)
            ```
4. Create venv and pip freeze the requirements.txt file
```python
    cd <project folder>
    python -m venv venv
    venv\Scripts\activate
    pip install pandas
    pip install fastapi
        ..
        ..
        ..
    pip install uvicorn
    pip list
    pip freeze > requirements.txt
```

5. Create Procfile
```python
   web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```
6. Repository is ready, to create an app on Herkou, and connect it with your github repository

7. Once that is done, `Deploy branch`

8. Check Heroku logs to make sure there are no errors. If there are no errors, go to https://predict-iris-flower-species.herokuapp.com/docs, 
  

## Author
[Akshit Agarwal](https://www.linkedin.com/in/akshit-agarwal93/)

