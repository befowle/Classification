## import required libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
from statsmodels.formula.api import ols
import statsmodels as sm
import statsmodels.api as sm
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from yellowbrick.classifier import classification_report
pd.set_option('display.max_columns', None)

#map dummies to 0's and 1's instead of 1's and 2's
map_dummies = {'Gender': {1: '0', 2: '1'},
              'Fever': {1: '0', 2: '1'},
              'Nausea/Vomting': {1: '0', 2: '1'},
              'Headache': {1: '0', 2: '1'},
              'Diarrhea': {1: '0', 2: '1'},
              'Fatigue & generalized bone ache': {1: '0', 2: '1'},
              'Jaundice': {1: '0', 2: '1'},
              'Epigastric pain': {1: '0', 2: '1'}}

data.replace(map_dummies, inplace=True)

#create plot to illustrate class balance
ax = sns.countplot(x="Transplant", data=data)

#view class counts
data['Transplant'].value_counts()

#change datatypes
df_edit['ALT 36'] = df_edit['ALT 36'].astype('int64')
df_edit['ALT 48'] = df_edit['ALT 48'].astype('int64')
df_edit['ALT after 24 w'] = df_edit['ALT after 24 w'].astype('int64')
df_edit['RNA 4'] = df_edit['RNA 4'].astype('int64')
df_edit['RNA 12'] = df_edit['RNA 12'].astype('int64')
df_edit['RNA EOT'] = df_edit['RNA EOT'].astype('int64')
df_edit['RNA EF'] = df_edit['RNA EF'].astype('int64')
df_edit['RNA 4'] = df_edit['RNA 4'].astype('int64')
df_edit['Baseline histological Grading'] = df_edit['Baseline histological Grading'].astype('int64') #re-replace with non-median values
df_edit.info()

#save as CSV
df_edit.to_csv('HepCDF.csv')

data = pd.read_csv('HepCDF.csv')

data.drop(['Unnamed: 0'], axis = 1, inplace=True)

data.to_csv('HepCDF.csv')

#compute correlation matrix
corr = data.corr()

#this masks the upper triangle of the heatmap
mask = np.triu(np.ones_like(corr, dtype=np.bool))

#this sets up the figure
f, ax = plt.subplots(figsize=(11, 9))

#this is a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmin= -0.3, vmax=0.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#change datatypes
data['Age'] = data['Age'].astype('float64')
data['Gender'] = data['Gender'].astype('object')
data['Fever'] = data['Fever'].astype('object')
data['Nausea/Vomting'] = data['Nausea/Vomting'].astype('object')
data['Headache'] = data['Headache'].astype('object')
data['Diarrhea'] = data['Diarrhea'].astype('object')
data['Fatigue & generalized bone ache'] = data['Fatigue & generalized bone ache'].astype('object')
data['Jaundice'] = data['Jaundice'].astype('object')
data['Epigastric pain'] = data['Epigastric pain'].astype('object')
data['Baseline histological Grading'] = data['Baseline histological Grading'].astype('object')
data['Transplant'] = data['Transplant'].astype('object')
data['ALT 36'] = data['ALT 36'].astype('float64')
data['ALT 48'] = data['ALT 48'].astype('float64')
data['ALT after 24 w'] = data['ALT after 24 w'].astype('float64')
data['RNA 4'] = data['RNA 4'].astype('float64')
data['RNA 12'] = data['RNA 12'].astype('float64')
data['RNA EOT'] = data['RNA EOT'].astype('float64')
data['RNA EF'] = data['RNA EF'].astype('float64')
data['RNA 4'] = data['RNA 4'].astype('float64')
data['BMI'] = data['BMI'].astype('float64')
data['WBC'] = data['WBC'].astype('float64')
data['RBC'] = data['RBC'].astype('float64')
data['HGB'] = data['HGB'].astype('float64')
data['Plat'] = data['Plat'].astype('float64')
data['AST 1'] = data['AST 1'].astype('float64')
data['ALT 1'] = data['ALT 1'].astype('float64')
data['ALT4'] = data['ALT4'].astype('float64')
data['ALT 12'] = data['ALT 12'].astype('float64')
data['ALT 24'] = data['ALT 24'].astype('float64')
data['RNA Base'] = data['RNA Base'].astype('float64')

#calculate z-scores
numeric_cols = data.select_dtypes(include=[np.number]).columns
z_scores = data[numeric_cols].apply(zscore)
z_scores

#create baseline model
X = data
y = transplant
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X, y)
DummyClassifier(strategy='stratified')
dummy_clf.predict(X)

dummy_clf.score(X, y)

#begin K-Nearest Neighbor model
#create KNN class
class KNN:
    def fit():
        pass
    def predict():
        pass

#complete the fit() method:
# self = instance method of KNN class
#X_train: array, each row is a vector of a given point in space
# y_train: corresponding labels for each vector in X_train
def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

# This line updates the knn.fit method to point to the function you've just written
KNN.fit = fit

#helper function: creates empty array to hold euclidean distance between x and X_Train
#Creates tuple using index and the distance, append to array
#return the distance arrray when distance has been generated for all items in self.X_train
def _get_distances(self, x):
    distances = []
    for ind, val in enumerate(self.X_train):
        dist_to_i = euclidean(x, val)
        distances.append((ind, dist_to_i))
    return distances

# This line attaches the function you just created as a method to KNN class
KNN._get_distances = _get_distances

#function to retreive indices of k-nearest points
#3 arguments: 1) self
# 2)dists: an array of tuples containing (index, distance), from _get_distances()
# 3) k: the number of distances you want to return
#sort dists array by values (the second element in each tuple)
#return first k tuples from the sorted array
def _get_k_nearest(self, dists, k):
    sorted_dists = sorted(dists, key=lambda x: x[1])
    return sorted_dists[:k]

# This line attaches the function you just created as a method to KNN class
KNN._get_k_nearest = _get_k_nearest

#begin classification tree
X_data=pd.DataFrame(data, columns=['Age', 'Gender', 'BMI', 'Fever', 'Nausea/Vomting', 'Headache',
       'Diarrhea', 'Fatigue & generalized bone ache', 'Jaundice',
       'Epigastric pain', 'WBC', 'RBC', 'HGB', 'Plat', 'AST 1', 'ALT 1',
       'ALT4', 'ALT 12', 'ALT 24', 'ALT 36', 'ALT 48', 'ALT after 24 w',
       'RNA Base', 'RNA 4', 'RNA 12', 'RNA EOT', 'RNA EF',
       'Baseline histological Grading'])
y_data=data.Transplant

y_data = y.astype('int')

X_train_data,X_test_data,y_train_data,y_test_data = train_test_split(X_data, y_data, test_size = 0.2, random_state = 42)

ctree=DecisionTreeClassifier(max_depth = 2)
ctree.fit(X_train_data,y_train_data)

dot_data = StringIO()
export_graphviz(ctree, out_file=dot_data,
                rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

clasPred = ctree.predict(X_test_data)

accuracy_score(y_test_data, clasPred)

#find optimal depth of the tree
from sklearn.model_selection import cross_val_score
score = cross_val_score(ctree, X_data, y_data, cv = 10)
score.mean()
depth_range = range(1,10)
val = []
for depth in depth_range:
    ctree = DecisionTreeClassifier(max_depth = depth)
    depth_score = cross_val_score(ctree, X_data, y_data, cv = 10)
    val.append(depth_score.mean())
print(val)
plt.figure(figsize = (10,10))
plt.plot(depth_range, val)
plt.xlabel('range of depth')
plt.ylabel('cross validated values')
plt.show()

#create bagged trees for Random Forest
#split outcome and predictor variables
transplant = data['Transplant']
data = data.drop('Transplant', axis = 1)
transplant = transplant.astype('int')

#create dummy columns with categorical variables
data = pd.get_dummies(data)
data.tail()

data_train, data_test, transplant_train, transplant_test = train_test_split(data, transplant,
                                                                    test_size = 0.25, random_state=123)

# Instantiate and fit a DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
tree_clf.fit(data_train, transplant_train)

# Feature importance
tree_clf.feature_importances_

#train feature importance
def plot_feature_importances(model):
    n_features = data_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')

plot_feature_importances(tree_clf)

# Test set predictions for bootstrapped
pred = tree_clf.predict(data_test)

# Confusion matrix and classification report
print(confusion_matrix(transplant_test, pred))
print(classification_report(transplant_test, pred))

print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(accuracy_score(transplant_test, pred) * 100))

# Instantiate a BaggingClassifier (changed maxdepth to 3 from 5)
bagged_tree =  BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=3),
                                 n_estimators=20)

# Fit to the training data
bagged_tree.fit(data_train, transplant_train)

#check accuracy score
# Training accuracy score
bagged_tree.score(data_train, transplant_train)

# Test accuracy score
bagged_tree.score(data_test, transplant_test)

#begin Random Forest model
forest = RandomForestClassifier(n_estimators=100, max_depth= 3)
forest.fit(data_train, transplant_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=3, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

# Training accuracy score
forest.score(data_train, transplant_train)

# Test accuracy score
forest.score(data_test, transplant_test)

plot_feature_importances(forest)

#create forest with small trees (max depth = 2)
# Instantiate and fit a RandomForestClassifier
forest_2 = RandomForestClassifier(n_estimators = 5, max_features= 10, max_depth= 2)
forest_2.fit(data_train, transplant_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=2, max_features=10, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=5,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

#adjust max_features (use fewer). store in .estimators_ attribute
#get the first tree and store it
# First tree from forest_2
rf_tree_1 = forest_2.estimators_[0]

#these are the features given to the tree during subspace sampling
# Feature importance of tree 1 from forest 2
plot_feature_importances(rf_tree_1)

# Second tree from forest_2
rf_tree_2 = forest_2.estimators_[1]

# Feature importance
plot_feature_importances(rf_tree_2)

from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import classification_report

#create classification report
X = data
y = transplant
visualizer = classification_report
    RandomForestClassifier(n_estimators=20), X, y
)

#run GridSearchCV to find optimal parameters for decision tree
y = data['Transplant']
X = data.drop('Transplant', axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

dt_clf = DecisionTreeClassifier()

dt_cv_score = cross_val_score(dt_clf, X_train, y_train, cv=3)
mean_dt_cv_score = np.mean(dt_cv_score)

print(f"Mean Cross Validation Score: {mean_dt_cv_score :.2%}")

dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]
}

# Instantiate GridSearchCV
dt_grid_search = GridSearchCV(dt_clf, dt_param_grid, cv=3, return_train_score=True)

# Fit to the data
dt_grid_search.fit(X_train, y_train)

# Mean training score
dt_gs_training_score = np.mean(dt_grid_search.cv_results_['mean_train_score'])

# Mean test score
dt_gs_testing_score = dt_grid_search.score(X_test, y_test)

print(f"Mean Training Score: {dt_gs_training_score :.2%}")
print(f"Mean Test Score: {dt_gs_testing_score :.2%}")
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_

#GridSearchCV optimized random forest model
#baseline score for Random Forest
rf_clf = RandomForestClassifier()
mean_rf_cv_score = np.mean(cross_val_score(rf_clf, data_train, transplant_train, cv=3))

print(f"Mean Cross Validation Score for Random Forest Classifier: {mean_rf_cv_score :.2%}")

rf_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]
}

#begin grid search
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=3)
rf_grid_search.fit(data_train, transplant_train)

#Training Accuracy: 74.37%
#Optimal Parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 5}

forest =RandomForestClassifier(
    criterion ='entropy',
    max_depth = 6,
    min_samples_leaf= 1,
    min_samples_split= 5)
forest.fit(data_train, transplant_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=6, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

forest.score(data_train, transplant_train)

forest.score(data_test, transplant_test)

plot_feature_importances(forest)

# Test set predictions for bootstrapped
pred = forest.predict(data_test)

# Confusion matrix and classification report
print(confusion_matrix(transplant_test, pred))
print(classification_report(transplant_test, pred))

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(data.shape[1]):
    print("%d. %s (%f)" % (f + 1, data.columns.values[indices[f]], importances[indices[f]]))

def plot_feature_importances(model):
    n_features = data_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')

plot_feature_importances(tree_clf)    
