from sklearn.metrics import roc_auc_score, accuracy_score

precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
acc_scores = []
error_rate_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    tree = DecisionTreeClassifier()

    tree.fit(X_train, y_train)

    predict = tree.predict(X_val)

    tp = sum((y_val == 1) & (predictions == 1))
    fp = sum((y_val == 0) & (predictions == 1))
    fn = sum((y_val == 1) & (predictions == 0))
    tn = sum((y_val == 0) & (predictions == 0))    


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(y_val, predictions)
    accuracy = accuracy_score(y_val, predictions)
    error_rate = 1 - accuracy

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)   
    auc_scores.append(auc)
    acc_scores.append(accuracy)
    error_rate_scores.append(error_rate)

average_precision = sum(precision_scores) / k
average_recall = sum(recall_scores) / k
average_f1 = sum(f1_scores) / k
average_auc = sum(auc_scores) / k
average_acc = sum(acc_scores) / k
average_error = sum(error_rate_scores) / k

print("Average Precision of Decision tree:", average_precision)
print("Average Recall of Decision tree:", average_recall)
print("Average F1-score of Decision tree:", average_f1)
print("Average AUC of Decision tree:", average_auc)
print("Average accuracy of Decision tree:", average_acc)
print("Average average error of Decision tree:", average_error)



precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
acc_scores = []
error_rate_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    bagg = BaggingClassifier()

    bagg.fit(X_train, y_train)

    predict = bagg.predict(X_val)

    tp = sum((y_val == 1) & (predictions == 1))
    fp = sum((y_val == 0) & (predictions == 1))
    fn = sum((y_val == 1) & (predictions == 0))
    tn = sum((y_val == 0) & (predictions == 0))    


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(y_val, predictions)
    accuracy = accuracy_score(y_val, predictions)
    error_rate = 1 - accuracy

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)   
    auc_scores.append(auc)
    acc_scores.append(accuracy)
    error_rate_scores.append(error_rate)

average_precision = sum(precision_scores) / k
average_recall = sum(recall_scores) / k
average_f1 = sum(f1_scores) / k
average_auc = sum(auc_scores) / k
average_acc = sum(acc_scores) / k
average_error = sum(error_rate_scores) / k

print("Average Precision of Bagging:", average_precision)
print("Average Recall of Bagging :", average_recall)
print("Average F1-score of Bagging :", average_f1)
print("Average AUC of Bagging :", average_auc)
print("Average accuracy of Bagging :", average_acc)
print("Average average error of Bagging :", average_error)



precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
acc_scores = []
error_rate_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    SVM = SVC(kernel='linear', C=1.0)
    SVM.fit(X_train, y_train)

    predict = SVM.predict(X_val)

    tp = sum((y_val == 1) & (predictions == 1))
    fp = sum((y_val == 0) & (predictions == 1))
    fn = sum((y_val == 1) & (predictions == 0))
    tn = sum((y_val == 0) & (predictions == 0))    


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(y_val, predictions)
    accuracy = accuracy_score(y_val, predictions)
    error_rate = 1 - accuracy

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)   
    auc_scores.append(auc)
    acc_scores.append(accuracy)
    error_rate_scores.append(error_rate)

average_precision = sum(precision_scores) / k
average_recall = sum(recall_scores) / k
average_f1 = sum(f1_scores) / k
average_auc = sum(auc_scores) / k
average_acc = sum(acc_scores) / k
average_error = sum(error_rate_scores) / k

print("Average Precision of SVM:", average_precision)
print("Average Recall of SVM:", average_recall)
print("Average F1-score of SVM:", average_f1)
print("Average AUC of SVM:", average_auc)
print("Average accuracy of SVM:", average_acc)
print("Average average error of SVM:", average_error)


precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
acc_scores = []
error_rate_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    knn = KNeighborsClassifier(3)
    knn.fit(X_train, y_train)

    predict = knn.predict(X_val)

    tp = sum((y_val == 1) & (predictions == 1))
    fp = sum((y_val == 0) & (predictions == 1))
    fn = sum((y_val == 1) & (predictions == 0))
    tn = sum((y_val == 0) & (predictions == 0))    


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(y_val, predictions)
    accuracy = accuracy_score(y_val, predictions)
    error_rate = 1 - accuracy

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)   
    auc_scores.append(auc)
    acc_scores.append(accuracy)
    error_rate_scores.append(error_rate)

average_precision = sum(precision_scores) / k
average_recall = sum(recall_scores) / k
average_f1 = sum(f1_scores) / k
average_auc = sum(auc_scores) / k
average_acc = sum(acc_scores) / k
average_error = sum(error_rate_scores) / k

print("Average Precision of KNN:", average_precision)
print("Average Recall of KNN:", average_recall)
print("Average F1-score of KNN:", average_f1)
print("Average AUC of KNN:", average_auc)
print("Average accuracy of KNN:", average_acc)
print("Average average error of KNN:", average_error)
