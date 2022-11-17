# defined function to test the model
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from assets import object_cols, labelencoder
import streamlit as st
import matplotlib.pyplot as plt
from process_data import scale_data, one_hot, get_label, strip_spaces, get_intlabel
import pickle
import numpy as np
import pandas as pd


def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model


def get_results(model):
    data = pd.read_csv('Data/Major_Crime_Indicators.csv')
    data.drop(['reporteddate', 'occurrencedate', 'event_unique_id', 'Hood_ID', 'X', 'Y', 'Index_', 'ObjectId', ], axis = 1, inplace = True)
    data = data[data['Longitude'] != 0]
    data['mci_category'] = [get_intlabel(item) for item in data['mci_category']]
    data.dropna(inplace = True)
    for col in object_cols:
        data[col] = data[col].apply(strip_spaces)

    encoders = pickle.load(open(labelencoder, 'rb'))
    for col in object_cols:
        data[col] = encoders[col].transform(data[col])

    X = data.drop('mci_category', axis = 1)
    X = scale_data(X)
    

    y = data['mci_category']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    del data
    pred = model.predict(X_test)

    st.write(classification_report(y_test, pred))

    y_oneHot = one_hot(np.array(y_test).reshape(-1, 1))
    
    prob_pred = model.predict_proba(X_test)
    
    n_classes = y_oneHot.shape[1]
    macro_roc_auc_ovo = roc_auc_score(y_oneHot, prob_pred, multi_class="ovo",
                                        average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_oneHot, prob_pred, multi_class="ovo",
                                        average="weighted")
    macro_roc_auc_ovr = roc_auc_score(y_oneHot, prob_pred, multi_class="ovr",
                                        average="macro")
    weighted_roc_auc_ovr = roc_auc_score(y_oneHot, prob_pred, multi_class="ovr",
                                        average="weighted")
    st.write("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)"
            .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    st.write("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted by prevalence)"
            .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
    

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_oneHot[:, i], prob_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        fig = plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'performance of the model {get_label(i)}')
        plt.legend(loc="lower right")
        st.pyplot(fig)

    del X_test, y_test, y_oneHot