import pickle
from assets import standerdscaler, onehotencoder, dict_unique_mci_category, dict_unique_mci_category_invert
import streamlit as st


def scale_data(data):
    scaler = pickle.load(open(standerdscaler, 'rb'))
    return scaler.transform(data)

def one_hot(y):
    encoder = pickle.load(open(onehotencoder, 'rb'))
    return encoder.transform(y).toarray()

def get_label(pred):
    return pickle.load(open(dict_unique_mci_category_invert, 'rb'))[pred]

def get_intlabel(y):
    return pickle.load(open(dict_unique_mci_category, 'rb'))[y]

def strip_spaces(x):
    return x.strip()

