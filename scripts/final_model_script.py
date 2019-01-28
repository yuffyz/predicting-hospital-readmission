import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics, svm, preprocessing, model_selection
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# define models
log_reg = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='newton-cg',
          tol=0.0001, verbose=0, warm_start=False)
multinomial_nb = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=False)
gaussian_nb = GaussianNB()
gradient_boosting = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=5, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=10,
              n_iter_no_change=None, presort='auto', random_state=None,
              subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)
bagging = BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
            criterion='gini', max_depth=8, max_features='log2',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=340,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators='warn', n_jobs=None, oob_score=False,
            random_state=12, verbose=0, warm_start=False),
         bootstrap=True, bootstrap_features=False, max_features=1.0,
         max_samples=1.0, n_estimators=15, n_jobs=None, oob_score=False,
         random_state=None, verbose=0, warm_start=False)
adaboost = AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight='balanced', criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=20,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=20, n_estimators=25, random_state=None)
random_forest = RandomForestClassifier(random_state=12, class_weight='balanced_subsample', criterion='gini',
max_depth=8, max_features='log2', min_samples_leaf=340)


def ensemble_cv(X,Y,model_list,KFold_splits,voting_type):
    kf = KFold(n_splits = KFold_splits, random_state = 12, shuffle = True) # shuffle True or False?
    
    iteration = 0
    test_accuracy_score_list = []
    train_accuracy_score_list = []
    train_recall_score_list = []
    test_recall_score_list = []
    test_precision_score_list = []
    train_precision_score_list = []
    test_f1_score_list = []
    train_f1_score_list = []
    class_0_accuracy_list = []
    class_1_accuracy_list = []
    class_0_recall_list = []
    class_1_recall_list = []
    train_roc_auc_list = []
    test_roc_auc_list = []
    confusion_matrix_list = []
    
    for train_index, test_index in kf.split(X):
        iteration += 1
        print("fold: ",iteration)
        
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = Y[train_index], Y[test_index]
        
        #scaler = preprocessing.MinMaxScaler()
        #train_X = scaler.fit_transform(train_X.astype(np.float64))
        #test_X = scaler.transform(test_X.astype(np.float64))

        #sm = RandomUnderSampler(random_state=12, ratio = 1.0)
        #train_X, train_Y = sm.fit_sample(train_X, train_Y)

        estimator_list = []
        model_names = ['m'+str(i+1) for i in range(len(model_list))]
        for model,model_name in zip(model_list,model_names):
            estimator_list.append((model_name,model))
        eclf = VotingClassifier(estimators=estimator_list, voting=voting_type)
        eclf.fit(train_X,train_Y)
        train_predictions = eclf.predict(train_X)
        test_predictions = eclf.predict(test_X)

        train_accuracy_score_list.append(metrics.accuracy_score(train_Y,train_predictions))
        test_accuracy_score_list.append(metrics.accuracy_score(test_Y,test_predictions))
        train_recall_score_list.append(metrics.recall_score(train_Y,train_predictions))
        test_recall_score_list.append(metrics.recall_score(test_Y,test_predictions))
        train_precision_score_list.append(metrics.precision_score(train_Y, train_predictions))
        test_precision_score_list.append(metrics.precision_score(test_Y, test_predictions))
        train_f1_score_list.append(metrics.f1_score(train_Y, train_predictions))
        test_f1_score_list.append(metrics.f1_score(test_Y, test_predictions))
        cm = metrics.confusion_matrix(test_Y,  test_predictions)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_0_accuracy_list.append(cm.diagonal()[0])
        class_1_accuracy_list.append(cm.diagonal()[1])
        class_0_recall_list.append(metrics.recall_score(test_Y,test_predictions,average=None)[0])
        class_1_recall_list.append(metrics.recall_score(test_Y,test_predictions,average=None)[1])
        train_roc_auc_list.append(metrics.roc_auc_score(train_Y, train_predictions))
        test_roc_auc_list.append(metrics.roc_auc_score(test_Y, test_predictions))
        
        confusion_matrix_list.append(metrics.confusion_matrix(test_Y, test_predictions))
        
    results_df = pd.DataFrame(index=['accuracy','recall','precision','f1','roc'])
    results_df['train'] = [np.mean(train_accuracy_score_list),np.mean(train_recall_score_list),
                           np.mean(train_precision_score_list),np.mean(train_f1_score_list),
                          np.mean(train_roc_auc_list)]
    results_df['test'] = [np.mean(test_accuracy_score_list),np.mean(test_recall_score_list),
                           np.mean(test_precision_score_list),np.mean(test_f1_score_list),
                          np.mean(test_roc_auc_list)]
    results_df['class_0_test'] = [np.mean(class_0_accuracy_list),np.mean(class_0_recall_list),'-','-','-']
    results_df['class_1_test'] = [np.mean(class_1_accuracy_list),np.mean(class_1_recall_list),'-','-','-']
    confusion_matrix = pd.DataFrame(index=['actual_class_0','actual_class_1'],columns=['predicted_class_0','predicted_class_1'])
    confusion_matrix.iloc[0] = sum(confusion_matrix_list)[0]
    confusion_matrix.iloc[1] = sum(confusion_matrix_list)[1]
    print(confusion_matrix)
    print(results_df)

# baseline hard voting ['multinomial_nb',gaussian_nb',gradient_boosting',random_forest']
# baseline hard voting ['multinomial_nb',adaboost',random_forest']
# baseline soft voting ['log_reg',multinomial_nb',gaussian_nb',bagging',adaboost',random_forest']

# feature selection based on RF feature selection
keep_features = ['number_inpatient', 'number_emergency', 'time_in_hospital', 
                 'num_medicine_change', 'number_diagnoses', 'insulin', 'discharge_diposition_expired', 
                 'num_total_medicine', 'A1Cresult', 'metformin', 'number_outpatient', 'age', 'diabetesMed', 
                 'num_lab_procedures', 'admission_source_emergency_room', 'num_procedures', 'diag1_circulatory', 
                 'num_medications', 'admission_emergency_urgent', 'diag1_symptoms', 'diag1_endocrine', 'is_male', 
                 'discharge_diposition_home_other_facility', 'admission_elective', 'rosiglitazone', 
                 'change_in_medications', 'race_aa', 'glyburide', 
                 'admission_source_transfer_hospital_health_care_facility_clinic', 'diag1_musculoskeletal', 
                 'admission_source_physician_referral', 'race_white', 'diag1_injury_poisoning', 'admission_unknown', 
                 'glimepiride', 'glipizide', 'pioglitazone', 'diag1_other', 'diag1_respiratory','readmitted']

# running chosen ensemble model, 10-fold CV
df = pd.read_csv("clean_diabetic_data.csv")
df.drop(['encounter_id','patient_id','category_diag_1'],axis=1,inplace=True)
df = df[keep_features]
X = np.array(df[df.columns[:-1]])
Y = np.array(df[df.columns[-1:]]['readmitted'])
chosen_ensemble = [multinomial_nb,gradient_boosting,random_forest]
print('Running 10-Fold CV on the chosen ensemble model...')
ensemble_cv(X,Y,chosen_ensemble,10,'hard')