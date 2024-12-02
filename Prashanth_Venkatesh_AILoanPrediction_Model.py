import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score

#Insert downloaded file path below
df = pd.read_csv('File_Path')

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Area'] = df['Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2 })
df['Status'] = df['Status'].map({'Y': 1, 'N': 0})
df['Dependents'] = df['Dependents'].map({'3+': 3, '1': 1, '2': 2, '0': 0}) #3+ is 3 or more, 'Dependents'
feature_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Area']  # Columns for X
target_column = 'Status'  # Column for Y


#Remove rows with NaN values in any column (features or target)
data_cleaned = df.dropna()
X_cleaned = data_cleaned[feature_columns].values.tolist()
Y_cleaned = data_cleaned[target_column].tolist()

# Create X and Y as lists
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, Y_cleaned, test_size=0.33)
X_validate, X_finaltest, y_validate, y_finaltest = train_test_split(X_test, y_test, test_size=0.5)


#Variables required for hyperparameter tuning
acc = 0
rec = 0
pre = 0
avgacc = 0
avgrec = 0
avgpre = 0
Y_pred_tuning = 0

#Decision Tree Tuning
maxacc_decision = 0
maxrec_decision = 0
maxpre_decision = 0
maxaccx_decision = 0
maxrecx_decision = 0
maxprex_decision = 0
#Default initialization
decisiontree = DecisionTreeClassifier(max_depth=1)
for x in range (1,1000):
        sumacc = 0
        sumrec = 0
        sumpre = 0
        decisiontree = DecisionTreeClassifier(max_depth=x)
        decisiontree.fit(X_train, y_train)
        for z in range (100):
                Y_pred_tuning = decisiontree.predict(X_validate)
                acc = accuracy_score(y_validate, Y_pred_tuning)
                sumacc += acc
                pre = precision_score(y_validate, Y_pred_tuning, average='binary')
                sumpre += pre
                rec = recall_score(y_validate, Y_pred_tuning, average='binary')
                sumrec += rec
        avgacc = sumacc/100
        avgpre = sumpre/100
        avgrec = sumrec/100
        if avgacc > maxacc_decision:
                maxacc_decision = avgacc
                maxaccx_decision = x
        if avgrec > maxrec_decision:
                maxrec_decision = avgrec
                maxrecx_decision = x
        if avgpre > maxpre_decision:
                maxpre_decision = avgpre
                maxprex_decision = x
print("\n\n\nDecision Tree")
print("\nHighest Testing Accuracy: " + str(round(maxacc_decision,2)) + " at parameter value: " + str(maxaccx_decision))
print("Highest Testing Precision: " + str(round(maxpre_decision,2)) + " at parameter value: " + str(maxprex_decision))
print("Highest Testing Recall: " + str(round(maxrec_decision,2)) + " at parameter value: " + str(maxrecx_decision))
print("\nBest Hyperparameter Value: " + str(round(maxaccx_decision)))

#Random Forest Tuning
maxacc_random = 0
maxrec_random = 0
maxpre_random = 0
maxaccx_random = 0
maxrecx_random = 0
maxprex_random = 0
#Default initialization
randomforest = RandomForestClassifier(max_depth=x)
for x in range (1,1000):
        sumacc = 0
        sumrec = 0
        sumpre = 0
        randomforest = RandomForestClassifier(max_depth=x)
        randomforest.fit(X_train, y_train)
        for z in range (100):
                Y_pred_tuning = randomforest.predict(X_validate)
                acc = accuracy_score(y_validate, Y_pred_tuning)
                sumacc += acc
                pre = precision_score(y_validate, Y_pred_tuning, average='binary')
                sumpre += pre
                rec = recall_score(y_validate, Y_pred_tuning, average='binary')
                sumrec += rec
        avgacc = sumacc/100
        avgpre = sumpre/100
        avgrec = sumrec/100
        if avgacc > maxacc_random:
                maxacc_random = avgacc
                maxaccx_random = x
        if avgrec > maxrec_random:
                maxrec_random = avgrec
                maxrecx_random = x
        if avgpre > maxpre_random:
                maxpre_random = avgpre
                maxprex_random = x
print("\n\n\nRandom Forest")
print("\nHighest Testing Accuracy: " + str(round(maxacc_random,2)) + " at parameter value: " + str(maxaccx_random))
print("Highest Testing Precision: " + str(round(maxpre_random,2)) + " at parameter value: " + str(maxprex_random))
print("Highest Testing Recall: " + str(round(maxrec_random,2)) + " at parameter value: " + str(maxrecx_random))
print("\nBest Hyperparameter Value: " + str(round(maxaccx_random)))



#MLP Tuning
maxaccx_MLP = 0
maxrecx_MLP = 0
maxprex_MLP = 0
maxacc_MLP = 0
maxrec_MLP = 0
maxpre_MLP = 0
#Default initialization
MLP = MLPClassifier(max_iter=300)
for x in range (300,1000):
        sumacc = 0
        sumrec = 0
        sumpre = 0
        MLP = MLPClassifier(max_iter=x)
        MLP.fit(X_train, y_train)
        for z in range (100):
                Y_pred_tuning = MLP.predict(X_validate)
                acc = accuracy_score(y_validate, Y_pred_tuning)
                sumacc += acc
                pre = precision_score(y_validate, Y_pred_tuning, average='binary')
                sumpre += pre
                rec = recall_score(y_validate, Y_pred_tuning, average='binary')
                sumrec += rec
        avgacc = sumacc/100
        avgpre = sumpre/100
        avgrec = sumrec/100
        if avgacc > maxacc_MLP:
                maxacc_MLP = avgacc
                maxaccx_MLP = x
        if avgrec > maxrec_MLP:
                maxrec_MLP = avgrec
                maxrecx_MLP = x
        if avgpre > maxpre_MLP:
                maxpre_MLP = avgpre
                maxprex_MLP = x
print("\n\n\nMLP")
print("\nHighest Testing Accuracy: " + str(round(maxacc_MLP,2)) + " at parameter value: " + str(maxaccx_MLP))
print("Highest Testing Precision: " + str(round(maxpre_MLP,2)) + " at parameter value: " + str(maxprex_MLP))
print("Highest Testing Recall: " + str(round(maxrec_MLP,2)) + " at parameter value: " + str(maxrecx_MLP))
print("\nBest Hyperparameter Value: " + str(round(maxaccx_MLP)))



#KNN Hyperparameter tuning
maxaccx_KNN = 0
maxrecx_KNN = 0
maxprex_KNN = 0
maxacc_KNN = 0
maxrec_KNN = 0
maxpre_KNN = 0
weightideal1 = ''
weightideal2 = ''
weightideal3 = ''
weights1 = 'uniform'
KNN = KNeighborsClassifier(n_neighbors=3, weights='uniform')
#Default initialization
for x in range (1,334):
        sumacc = 0
        sumrec = 0
        sumpre = 0
        KNN = KNeighborsClassifier(n_neighbors=x, weights=weights1)
        KNN.fit(X_train, y_train)
        for z in range (100):
                Y_pred_tuning = KNN.predict(X_validate)
                acc = accuracy_score(y_validate, Y_pred_tuning)
                sumacc += acc
                pre = precision_score(y_validate, Y_pred_tuning, average='binary')
                sumpre += pre
                rec = recall_score(y_validate, Y_pred_tuning, average='binary')
                sumrec += rec
        avgacc = sumacc/100
        avgpre = sumpre/100
        avgrec = sumrec/100
        if avgacc > maxacc_KNN:
                maxacc_KNN = avgacc
                maxaccx_KNN = x
                weightideal1 = 'uniform'
        if avgrec > maxrec_KNN:
                maxrec_KNN = avgrec
                maxrecx_KNN = x
                weightideal2 = 'uniform'
        if avgpre > maxpre_KNN:
                maxpre_KNN = avgpre
                maxprex_KNN = x
                weightideal3 = 'uniform'

weights1 = 'distance'
for x in range (1,334):
        sumacc = 0
        sumrec = 0
        sumpre = 0
        KNN = KNeighborsClassifier(n_neighbors=x, weights=weights1)
        KNN.fit(X_train, y_train)
        for z in range (100):
                Y_pred_tuning = KNN.predict(X_validate)
                acc = accuracy_score(y_validate, Y_pred_tuning)
                sumacc += acc
                pre = precision_score(y_validate, Y_pred_tuning, average='binary')
                sumpre += pre
                rec = recall_score(y_validate, Y_pred_tuning, average='binary')
                sumrec += rec
        avgacc = sumacc/100
        avgpre = sumpre/100
        avgrec = sumrec/100
        if avgacc > maxacc_KNN:
                maxacc_KNN = avgacc
                maxaccx_KNN = x
                weightideal1 = 'uniform'
        if avgrec > maxrec_KNN:
                maxrec_KNN = avgrec
                maxrecx_KNN = x
                weightideal2 = 'uniform'
        if avgpre > maxpre_KNN:
                maxpre_KNN = avgpre
                maxpre_KNNx = x
                weightideal3 = 'uniform'

print("\n\n\nKNN")
print("\nHighest Testing Accuracy: " + str(round(maxacc_KNN,2)) + " at parameter value: " + str(maxaccx_KNN) + " and with " + str(weightideal1) + " weights")
print("Highest Testing Precision: " + str(round(maxpre_KNN,2)) + " at parameter value: " + str(maxprex_KNN) + " and with " + str(weightideal2) + " weights")
print("Highest Testing Recall: " + str(round(maxrec_KNN,2)) + " at parameter value: " + str(maxrecx_KNN) + " and with " + str(weightideal3) + " weights")
count = 0
if weightideal1 == 'uniform':
        count += 1
if weightideal2 == 'uniform':
        count += 1
if weightideal3 == 'uniform':
        count += 1
finalweight = ''
if count >=2:
        print("\nBest Hyperparameter Value: " + str(round(maxaccx_KNN)) + " and with 'uniform' weights")
        finalweight = 'uniform'
else:
        print("\nBest Hyperparameter Value: " + str(round(maxaccx_KNN)) + " and with 'distance' weights")
        finalweight = 'distance'

#Here the code puts the hyperparameters on its own without human intervention
#Instead one could simply input the ideal hyperparameters provided in the paper and comment out the hyperparameter testing code
Logistic = LogisticRegression()

decisiontree = DecisionTreeClassifier(max_depth=maxaccx_decision)

randomforest = RandomForestClassifier(max_depth=maxaccx_random)
#Random forest is a machine learning model that involves using several decision trees. At the end we take the majority from all the decision trees for the final predictions.

MLP = MLPClassifier(max_iter=maxaccx_MLP)
#Multi Layer Perceptron - Neural Network

KNN = KNeighborsClassifier(n_neighbors=maxaccx_KNN, weights=finalweight)
#K-Nearest Neighbour


#Weights Algoirthm
#Train the model using the training data
Logistic.fit(X_train, y_train)
decisiontree.fit(X_train, y_train)
randomforest.fit(X_train, y_train)
MLP.fit(X_train, y_train)
KNN.fit(X_train, y_train)

check = False
result = 0
sum = 0
weight_logistic = 1
weight_decision = 1
weight_forest = 1
weight_MLP = 1
weight_KNN = 1
count = 0
count1 = 0
for i in range(0,len(X_finaltest)):
    sum = 0
    Y_pred_logistic = Logistic.predict(np.array(X_finaltest[i]).reshape(1, -1))
    Y_pred_decision = decisiontree.predict(np.array(X_finaltest[i]).reshape(1, -1))
    Y_pred_forest = randomforest.predict(np.array(X_finaltest[i]).reshape(1, -1))
    Y_pred_MLP = MLP.predict(np.array(X_finaltest[i]).reshape(1, -1))
    Y_pred_KNN = KNN.predict(np.array(X_finaltest[i]).reshape(1, -1))
    

    print("KNN Prediction:" + str(Y_pred_KNN[0]))
    print("MLP Prediction:" + str(Y_pred_MLP[0]))
    print("forest Prediction:" + str(Y_pred_forest[0]))
    print("decision Prediction:" + str(Y_pred_decision[0]))
    print("logisitc Prediction:" + str(Y_pred_logistic[0]))


    if Y_pred_decision == y_finaltest[i]:
            weight_decision += 1
    if Y_pred_logistic == y_finaltest[i]:
            weight_logistic += 1
    if Y_pred_forest == y_finaltest[i]:
            weight_forest += 1
    if Y_pred_MLP == y_finaltest[i]:
            weight_MLP += 1
    if Y_pred_KNN == y_finaltest[i]:
            weight_KNN += 1

    sum = (weight_decision + weight_forest + weight_logistic + weight_MLP + weight_KNN)
    result = ((Y_pred_KNN[0] * weight_KNN) + (Y_pred_logistic[0] * weight_logistic) + (Y_pred_decision[0] * weight_decision) + (Y_pred_forest[0] * weight_forest) + (Y_pred_MLP[0] * weight_MLP) + (Y_pred_KNN[0] * weight_KNN))
    

    print("Sum: " + str(sum))
    print("Result:" + str(result))  

    
    if result > sum/2:
        result = 1
    else:
        result = 0


    print("Prediction: " + str(result))
    print("Correct Value: " + str(y_test[i]) + "\n\n")

    
    count1 += 1
    if result==y_finaltest[i] :
        count += 1
print("\n\nNo. of Correct Predictions: " + str(count))
print("No. of Total Predictions: " + str(count1))
print("Accuracy = " + str(round((count/count1),2)))

print("\n\nWeights of each model")
print("Logistic Weight: " + str(weight_logistic))
print("Forest Weight: " + str(weight_forest))
print("Decision Weight: " +str(weight_decision))
print("KNN Weight: " + str(weight_MLP))
print("MLP Weight: " + str(weight_KNN))



# Make predictions on the test data
Y_pred_logistic = Logistic.predict(X_finaltest)
Y_pred_decision = decisiontree.predict(X_finaltest)
Y_pred_forest = randomforest.predict(X_finaltest)
Y_pred_MLP = MLP.predict(X_finaltest)
Y_pred_KNN = KNN.predict(X_finaltest)


# Testing Accuracy
print("\n\nTesting Accuracy")
accuracy_logisitic = accuracy_score(y_finaltest, Y_pred_logistic)
print(f"Logistic Accuracy: {accuracy_logisitic:.2f}")

accuracy_decision = accuracy_score(y_finaltest, Y_pred_decision)
print(f"Decision Accuracy: {accuracy_decision:.2f}")

accuracy_forest = accuracy_score(y_finaltest, Y_pred_forest)
print(f"Forest Accuracy: {accuracy_forest:.2f}")

accuracy_MLP = accuracy_score(y_finaltest, Y_pred_MLP)
print(f"MLP Accuracy: {accuracy_MLP:.2f}")

accuracy_KNN = accuracy_score(y_finaltest, Y_pred_KNN)
print(f"KNN Accuracy: {accuracy_KNN:.2f}")


# Testing Precision
print("\n\nTesting Precision")
precision_logistic = precision_score(y_finaltest, Y_pred_logistic, average='binary')
print(f"Logistic Precision: {precision_logistic:.2f}")

precision_decision = precision_score(y_finaltest, Y_pred_decision, average='binary')
print(f"Decision Precision: {precision_decision:.2f}")

precision_forest = precision_score(y_finaltest, Y_pred_forest, average='binary')
print(f"Forest Precision: {precision_forest:.2f}")

precision_MLP = precision_score(y_finaltest, Y_pred_MLP, average='binary')
print(f"MLP Precision: {precision_MLP:.2f}")

precision_KNN = precision_score(y_finaltest, Y_pred_KNN, average='binary')
print(f"KNN Precision: {precision_KNN:.2f}")


# Testing Recall
print("\n\nTesting Recall")
recall_logistic = recall_score(y_finaltest, Y_pred_logistic, average='binary')
print(f"Logistic Recall: {recall_logistic:.2f}")

recall_decision = recall_score(y_finaltest, Y_pred_decision, average='binary')
print(f"Decision Recall: {recall_decision:.2f}")

recall_forest = recall_score(y_finaltest, Y_pred_forest, average='binary')
print(f"Forest Recall: {recall_forest:.2f}")

recall_MLP = recall_score(y_finaltest, Y_pred_MLP, average='binary')
print(f"MLP Recall: {recall_MLP:.2f}")

recall_KNN = recall_score(y_finaltest, Y_pred_KNN, average='binary')
print(f"KNN Recall: {recall_KNN:.2f}")



#Training Predictions
Y_pred_logistic_t = Logistic.predict(X_train)
Y_pred_decision_t = decisiontree.predict(X_train)
Y_pred_forest_t = randomforest.predict(X_train)
Y_pred_MLP_t = MLP.predict(X_train)
Y_pred_KNN_t = KNN.predict(X_train)

# Training Accuracy
print("\n\nTraining Accuracy")
accuracy_logisitic_t = accuracy_score(y_train, Y_pred_logistic_t)
print(f"Logistic Accuracy: {accuracy_logisitic_t:.2f}")

accuracy_decision_t = accuracy_score(y_train, Y_pred_decision_t)
print(f"Decision Accuracy: {accuracy_decision_t:.2f}")

accuracy_forest_t = accuracy_score(y_train, Y_pred_forest_t)
print(f"Forest Accuracy: {accuracy_forest_t:.2f}")

accuracy_MLP_t = accuracy_score(y_train, Y_pred_MLP_t)
print(f"MLP Accuracy: {accuracy_MLP_t:.2f}")

accuracy_KNN_t = accuracy_score(y_train, Y_pred_KNN_t)
print(f"KNN Accuracy: {accuracy_KNN_t:.2f}")
