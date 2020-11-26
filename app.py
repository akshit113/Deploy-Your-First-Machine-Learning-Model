import pickle
from iris import Iris
import uvicorn
from fastapi import FastAPI

app = FastAPI()
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'hello, stranger'}


@app.post('/predict')
def predict_flower(data: Iris):
    data = data.dict()
    sepal_length = data['sepal_width']
    sepal_width = data['sepal_length']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if (prediction[0] == 0):
        prediction = "Iris Veriscolor (0)"
    elif (prediction[0] == 1):
        prediction = "Iris Setosa (1)"
    elif (prediction[0] == 2):
        prediction = 'Iris Virginica (2)'
    else:
        prediction = 'Not a valid prediction'
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    print('done')
