import numpy as np
import pandas as pd
#import pickle

dataset=pd.read_csv('salary_predict_dataset.csv')

#DEALING WITH NULL VALUES
dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)
dataset['interview_score'].fillna(dataset['interview_score'].mean(),inplace=True)

#CONVERTING THE WORDS IN YEARS_EXPERIENCE INTO INTEGERS
def convert_to_integer(word):
    word_dict={'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0,'fifteen':15,'thirteen':13,'fourteen':14}
    return word_dict[word]
dataset['experience']=dataset['experience'].apply(lambda x: convert_to_integer(x))
#print(dataset)

#SPLITTING THE DATA INTO INPUTS AND TARGETS:
X=dataset.iloc[:,:3]
Y=dataset.iloc[:,-1]

#FIT OUR DATA INTO A MODEL
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,Y)

pred=reg.predict([[2,9,6]])
#print(pred)

#SAVING THE MODEL TO DISK
import pickle
pickle.dump(reg,open('model.pkl','wb'))

#TESTING USING THE SAVED MODEL BY IMPORTING IT
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]])) #WHICH IS SAME AS ABOVE


