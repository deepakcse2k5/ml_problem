from scipy.stats import alpha
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred):
    counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt==yp:
            counter+=1
        return counter/len(y_true)



# print(metrics.accuracy_score(l1,l2))
# print(accuracy(l1,l2))

def true_positive(y_true, y_pred):
    counter = 0
    for yt,yp in zip(y_true, y_pred):
        if yt==1 and yp==1:
            counter+=1
    return counter

def true_negative(y_true,y_pred):
    counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            counter += 1
    return counter

def false_positive(y_true, y_pred):
    counter = 0
    for yt,yp in zip(y_true, y_pred):
        if yt==0 and yp==1:
            counter+=1
    return counter

def false_negative(y_true,y_pred):
    counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            counter += 1
    return counter

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

# print(f"true positive:{true_positive(l1,l2)}")
# print(f"true negative:{true_negative(l1,l2)}")
# print(f"flase positive:{false_positive(l1,l2)}")
# print(f"flase negative:{false_negative(l1,l2)}")


def accuracy_v1(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    accuracy = round((tp+tn)/ (tp+fp+tn+fn),2)
    precision = round(tp/(tp+fp),2)
    recall = round(tp/(tp+fn),2)
    fpr = round(fp/(tn+fp),2)
    f1_score = round(2*precision*recall/(precision+recall),2)
    print(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}, False Positive Rate:{fpr}, F1-Score:{f1_score}")
    return  recall, fpr, f1_score,accuracy, precision,

print(f"Evaluation Metrics:{accuracy_v1(l1,l2)}")




# TPR and FPR

tpr_list = []
fpr_list = []

y_true = [0,0,0,0,1,0,1,0,0,1,0,1,0,0,1]

y_pred = [0.1,0.3,0.2,0.6,0.8,0.05,0.9,0.5,0.3,0.66,0.3,0.2,0.85,0.15,0.99]

thresholds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.99,1.0]

for threshold in thresholds:
    tem_pred = [1 if x>=threshold else 0 for x in y_pred]

    tem_tpr,tem_fpr,_,_,_ = accuracy_v1(y_true, y_pred)
    tpr_list.append(tem_tpr)
    fpr_list.append(tem_fpr)

plt.figure(figsize=(7,7))
plt.fill_between(fpr_list,tpr_list,alpha=0.4)
plt.plot(fpr_list,tpr_list, lw=3)
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel("FPR", fontsize=15)
plt.ylabel("TPR", fontsize=15)
plt.show()








