__author__ = 'Akshit Agarwal'
__email__ = 'akshit@email.arizona.edu'
__date__ = '2020-11-23'
__dataset__ = 'http://archive.ics.uci.edu/ml/datasets/Iris/'
__connect__ = 'https://www.linkedin.com/in/akshit-agarwal93/'


from pickle import dump

from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data():
    data = load_iris()
    df = DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    print(df)
    return df


def split_dataset(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def build_model_and_predict():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model


def calculate_accuracy(y_pred):
    score = accuracy_score(y_test, y_pred)
    return score


def save_model(model):
    pickle_out = open("classifier.pkl", "wb")
    dump(model, pickle_out)
    pickle_out.close()



if __name__ == '__main__':
    df = load_data()
    X_train, X_test, y_train, y_test = split_dataset(df)
    y_pred, model = build_model_and_predict()
    score = calculate_accuracy(y_pred)
    save_model(model)
    print(y_pred)

    print('test')
