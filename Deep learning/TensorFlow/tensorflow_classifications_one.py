import pandas as pd
import tensorflow as tf

diabetes=pd.read_csv('/home/mayank-s/PycharmProjects/myprojects/datasets/pima-indians-diabetes.csv')
# print (diabetes.head())
print (diabetes.columns)
my_list=['Number_pregnant','Glucose_concentration', 'Blood_pressure', 'Triceps','Insulin', 'BMI','Pedigree']
# print (my_data.head())

#one way to normallise datasets another method is there in skilearn standard scalers
diabetes[my_list]=diabetes[my_list].apply(lambda x:(x-x.min())/(x.max()-x.min()))


print (diabetes.head())
#now tensorflow environment

Num_preg=tf.feature_column.numeric_column('Number_pregnant')
GC=tf.feature_column.numeric_column('Glucose_concentration')
BP=tf.feature_column.numeric_column('Blood_pressure')
tri=tf.feature_column.numeric_column('Triceps')
Ins=tf.feature_column.numeric_column('Insulin')
Bmi=tf.feature_column.numeric_column( 'BMI')
ped=tf.feature_column.numeric_column('Pedigree')
age=tf.feature_column.numeric_column('Age')
assign_group=tf.feature_column.categorical_column_with_hash_bucket("Group",hash_bucket_size=5)
age_buckets=tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])

feat_column=[Num_preg,GC,BP,tri,Ins,Bmi,ped,assign_group,age_buckets]

features=diabetes.drop('Class',axis=1)
labels=diabetes['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = .3, random_state = 0)


input_function=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=100,shuffle=True)

#this is for artificial neural network or single perceptron
'''model=tf.estimator.LinearClassifier(feature_columns=feat_column,n_classes=2)
model.train(input_fn=input_function,steps=1000)
print (input_function)
evaluation=tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
print (model.evaluate(evaluation))'''

#lets start for dense neural network
embedding_group=tf.feature_column.embedding_column(assign_group,dimension=4)
feat_column_2=[Num_preg,GC,BP,tri,Ins,Bmi,ped,embedding_group,age_buckets]

model_dnn=tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_column_2,n_classes=2)#hidden unit[10,10,10] means 3 hidden layer with 10 neuron each
model_dnn.train(input_function,steps=1000)
evaluation=tf.evaluation=tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
print(model_dnn.evaluate((evaluation)))
