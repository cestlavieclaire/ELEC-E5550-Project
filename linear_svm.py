from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

import csv, datetime


def doLinearSVC(X_train, X_test, y_train, y_test):
    #Cs = [0.001, 0.01, 0.1, 1, 10]
    #gammas = [0.001, 0.01, 0.1, 1]
    #param_grid = {'C': Cs}
    #clf= OneVsRestClassifier(GridSearchCV(LinearSVC(random_state=0), param_grid))
    
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    clf.fit(X_train, y_train)
    # predict and evaluate predictions
    predictions = clf.predict(X_test)
    draw_learning_curve(estimator=clf, title="LinearSVC", X=predictions, y=y_test)
    return clf, predictions

def doLogisticRegression(X_train, X_test, y_train, y_test):
    clf = OneVsRestClassifier(LogisticRegression(random_state=0))
    clf.fit(X_train, y_train)
    # predict and evaluate predictions
    predictions = clf.predict(X_test)
    draw_learning_curve(estimator=clf, title="LogisticRegression", X=predictions, y=y_test)
    return clf, predictions

def doLogisticRegressionCV(X_train, X_test, y_train, y_test):
    clf = OneVsRestClassifier(LogisticRegressionCV(random_state=0))
    clf.fit(X_train, y_train)
    # predict and evaluate predictions
    predictions = clf.predict(X_test)
    draw_learning_curve(estimator=clf, title="LogisticRegressionCV", X=predictions, y=y_test)
    return clf, predictions

def doSGDClassifier(X_train, X_test, y_train, y_test):
    clf = OneVsRestClassifier(SGDClassifier(random_state=0))
    clf.fit(X_train, y_train)
    # predict and evaluate predictions
    predictions = clf.predict(X_test)
    draw_learning_curve(estimator=clf, title="SGDClassifier", X=predictions, y=y_test)
    return clf, predictions

def doGaussianProcessClassifier(X_train, X_test, y_train, y_test):
    clf = OneVsRestClassifier(GaussianProcessClassifier(random_state=0))
    clf.fit(X_train, y_train)
    # predict and evaluate predictions
    predictions = clf.predict(X_test)
    draw_learning_curve(estimator=clf, title="GaussianProcessClassifier", X=predictions, y=y_test)
    return clf, predictions

def doPassiveAggressiveClassifier(X_train, X_test, y_train, y_test):
    clf = OneVsRestClassifier(PassiveAggressiveClassifier(random_state=0))
    clf.fit(X_train, y_train)
    # predict and evaluate predictions
    predictions = clf.predict(X_test)
    draw_learning_curve(estimator=clf, title="PassiveAggressiveClassifier", X=predictions, y=y_test)
    return clf, predictions

def doGradientBoostingClassifier(X_train, X_test, y_train, y_test):
    clf = OneVsRestClassifier(GradientBoostingClassifier(random_state=0))
    clf.fit(X_train, y_train)
    # predict and evaluate predictions
    predictions = clf.predict(X_test)
    draw_learning_curve(estimator=clf, title="GradientBoostingClassifier", X=predictions, y=y_test)
    return clf, predictions

def eval(X_train, y_train, y_test, clf, predictions, method):
    print(method,":", cross_val_score(clf, X_train, y_train, cv=2))
    print("Prediction: {}".format(np.array(predictions).shape))
    print("Prediction: {}".format(len(predictions)))
    print("Test labels: {}".format(np.array(y_test).shape))
    print("Test labels: {}".format(len(y_test)))
    print("Classification report for %s:\n%s\n" % (method, metrics.classification_report(y_test, predictions)))

def main():
    # This script concatenates all news headlines of a day into one and uses the tf-idf scheme to extract a feature vector.
    # An SVM with rbf kernel without optimization of hyperparameters is used as a classifier.
    
    # read data
    stopwords = [line.strip() for line in open("/u/32/tamperm1/unix/git/nlp-course-project/src/stopwords.txt.lst", 'r')]
    dcty = get_y_dataset('/u/32/tamperm1/unix/git/nlp-course-project/src/anonym_categories.csv')
    dataset = read_csv('/u/32/tamperm1/unix/git/nlp-course-project/data/document_abstract_data_lemmatized.csv', stopwords, dcty)
    y = list()
    # populate y
    print("populate y")
    yid = list(dataset.keys())
    for id in yid:
        if id in dcty:
            y.append(dcty[id])
            
    y_values = MultiLabelBinarizer().fit_transform(y)
    
    # calculate tf-idf for words in document
    x_all = np.array(list(dataset.values()))
    X = td_idf(x_all)
    
    # split into training- and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y_values, test_size=0.2, random_state=0)
    
    
    print("Train classifier")
    print("Training features: {}".format(np.array(X_train).shape))
    print("Traning features: {}".format(len(X_train)))
    print("Training labels: {}".format(np.array(y_train).shape))
    print("Traning labels: {}".format(len(y_train)))
    
    # train classifier
    
    clf, predictions = doLinearSVC(X_train, X_test, y_train, y_test)
    eval(X_train, y_train, y_test, clf, predictions, "LinearSVC")
    
    #clf, predictions = doLogisticRegression(X_train, X_test, y_train, y_test)
    #eval(X_train, y_train, y_test, clf, predictions, "LogisticRegression")
    
    #clf, predictions = doLogisticRegressionCV(X_train, X_test, y_train, y_test)
    #eval(X_train, y_train, y_test, clf, predictions, "LogisticRegressionCV")
    
    #clf, predictions = doSGDClassifier(X_train, X_test, y_train, y_test)
    #eval(X_train, y_train, y_test, clf, predictions, "SGDClassifier")
    
    #clf, predictions = doGaussianProcessClassifier(X_train, X_test, y_train, y_test)
    #eval(X_train, y_train, y_test, clf, predictions, "GaussianProcessClassifier")
    
    #clf, predictions = doGradientBoostingClassifier(X_train, X_test, y_train, y_test)
    #eval(X_train, y_train, y_test, clf, predictions, "doGradientBoostingClassifier")
    
    
def draw_learning_curve(estimator, title, X, y):

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(estimator, title, X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=4)
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    t = title.replace(" ", "_")
    filename = 'figs/plots/learning/' + t +'_learning_curve.png'
    fig.savefig(filename, dpi=fig.dpi)
    plt.cla()
    plt.close(fig)
    plt.close('all')
    
def td_idf(x_all):
    print("Calculate TF_IDF scores")
    
    tvec = TfidfVectorizer(analyzer=lambda d: d.split(','), lowercase=True).fit(x_all)

    tvec_weights_all = tvec.transform(x_all)
            
    return tvec_weights_all.toarray()

def read_csv( file, stopwords, y):
    data = dict()
    counter = 0
    print("Read csv data")
    
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            print(row[0])
            if row[0].split('.')[0] in y:                
                lst = list(filter(len, row[1:-1]))
                filtered = [word.lower() for word in lst if word not in stopwords and len(word.strip()) > 1 and clean_data(word)==True]
                data[row[0].split('.')[0]]=",".join(filtered)
                counter += 1
    print("Read ", counter, " documents")
    return data

def get_y_dataset(dataset):
    dct_y = dict()
    with open(dataset, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            dct_y[row[0].split('.')[0]] = list(filter(len, row[1:-1]))
            
    return dct_y

def clean_data(word):
    # if string contains only digits -> bad
    if word.isdigit()==True:
        return False
    
    # if string is a date
    if validate_date(word)==True:
        return False
    
    # if string contains alphabets, the string is good, otherwise has no value
    if any(c.isalpha() for c in word)==False:
        return False
    
    return True
            
    
def validate_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%d.%m.%Y')
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()
