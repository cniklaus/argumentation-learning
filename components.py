import pickle
import read_input_file
from sklearn.externals import joblib
import pandas as pd

dataframe = read_input_file.create_data_set(True, 0)
print(dataframe)

loaded_model = pickle.load(open("classifiers_components/Gaussian_NB_model.sav", 'rb'))

#y_pred = loaded_model.predict(dataframe)
#print(y_pred)


loaded_model.predict(dataframe)