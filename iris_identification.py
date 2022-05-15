"""
Me following a tutorial from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ 
"""
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import random


# Load dataset
IRIS_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
IRIS_NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
IRIS_DATA = read_csv(IRIS_URL, names=IRIS_NAMES)

# IRIS_DATA.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# IRIS_DATA.hist()
# pyplot.show()

# scatter_matrix(IRIS_DATA)
# pyplot.show()

data_array = IRIS_DATA.values
x_data = data_array[:,0:4]
y_data = data_array[:,4]

x_train, x_val, y_train, y_val = train_test_split(
    x_data, 
    y_data, 
    test_size=0.20, 
    random_state=random.randint(1, 100)
)

# k fold cross validation
# we split our data into 10, train on 9, test on 1, repeat for all train-test splits
# models = {
#     "LR": LogisticRegression(solver="liblinear", multi_class="ovr"),
#     "LDA": LinearDiscriminantAnalysis(),
#     "KNN": KNeighborsClassifier(),
#     "CART": DecisionTreeClassifier(),
#     "NB": GaussianNB(),
#     "SVM": SVC(gamma="auto"),
# }

# results = {}

# for name, model in models.items():
#     k_fold = StratifiedKFold(
#         n_splits=10, 
#         random_state=random.randint(1, 100), 
#         shuffle=True
#     )
#     cv_results = cross_val_score(model, x_train, y_train, cv=k_fold, scoring="accuracy")
#     results[name] = cv_results
#     print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

model = SVC(gamma="auto")
model.fit(x_train, y_train)
predictions = model.predict(x_val)
print(accuracy_score(y_val, predictions))
print(confusion_matrix(y_val, predictions))
print(classification_report(y_val, predictions))
