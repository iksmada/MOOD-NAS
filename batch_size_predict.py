import argparse

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, max_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import pickle

MODEL_NAME = 'batch_predict.pkl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train memory consumption predictor")
    parser.add_argument("-d", "--data", type=argparse.FileType('r'), required=True,
                        help="CSV file with columns 'model size' 'batch size' and 'GPU mb'")
    args = parser.parse_args()
    df = pd.read_csv(args.data)

    data = df[df.columns[:-1]].to_numpy()
    target = df[df.columns[-1]].to_numpy()
    reg = make_pipeline(PolynomialFeatures(2), linear_model.Ridge(alpha=0.5))
    reg.fit(data, target)
    pred = reg.predict(data)
    print("MAE = %.0f, MAX = %.0f, MAPE = %.0f%%" % (mean_absolute_error(target, pred), max_error(target, pred),
                                                     mean_absolute_percentage_error(target, pred)*100))
    print(list(zip(target, pred.astype(int))))

    # save the model to disk
    filename = MODEL_NAME
    pickle.dump(reg, open(filename, 'wb'))

    # load the model from disk to test
    loaded_model = pickle.load(open(filename, 'rb'))
    pred = loaded_model.predict(data)
    print("MAE = %s" % mean_absolute_error(target, pred))