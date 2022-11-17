
from create_models import load_model, get_results
from process_data import scale_data, get_label
from assets import column_names, mode_data, months, neighbourhood, object_cols
from assets import division, location_type, premises_type, ucr_code, ucr_ext, offence
import streamlit as st
import pandas as pd
import pickle
import datetime


def make_prediction(model_path):
    model = load_model(model_path)
    c11, c12 = st.columns([1, 1])
    c11.markdown('### Show model performance: ')
    if c12.checkbox(''):
        get_results(model)
    else:
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])
        reported_date = st.date_input('Enter Report Date', 
                                        datetime.date(int(mode_data[column_names[6]]), 
                                        months.index(mode_data[column_names[7]]), 
                                        int(mode_data[column_names[8]])))
        c7, c8, c9, c10, c11, c12 = st.columns([1, 1, 1, 1, 1, 1])
        occurence_date = st.date_input('Enter Occurence Date', 
                                        datetime.date(int(mode_data[column_names[6]]), 
                                        months.index(mode_data[column_names[7]]), 
                                        int(mode_data[column_names[8]])))
        c13, c14, c15, c16, c17, c18 = st.columns([1, 1, 1, 1, 1, 1])

        c19, c20, c21, c22, c23, c24 = st.columns([1, 1, 1, 1, 1, 1])
        st_columns = [c1, c2, c3, c4, c5, c6,
                    c7, c8, c9, c10, c11, c12,
                    c13, c14, c15, c16, c17, c18,
                    c19, c20, c21, c22, c23, c24]

        input_X = {a:None for a in column_names}


        input_X[column_names[0]] = [st_columns[0].selectbox(f'{column_names[0]}', division)]
        input_X[column_names[1]] = [st_columns[1].selectbox(f'{column_names[1]}', location_type)]
        input_X[column_names[2]] = [st_columns[2].selectbox(f'{column_names[2]}', premises_type)]
        input_X[column_names[3]] = [st_columns[3].selectbox(f'{column_names[3]}', ucr_code)]
        input_X[column_names[4]] = [st_columns[4].selectbox(f'{column_names[4]}', ucr_ext)]
        input_X[column_names[5]] = [st_columns[5].selectbox(f'{column_names[5]}', offence)]
        
        input_X[column_names[6]] = [st_columns[6].text_input(f'{column_names[6]}', value = reported_date.year, disabled = True)]
        input_X[column_names[7]] = [st_columns[7].text_input(f'{column_names[7]}', value = reported_date.strftime("%B"), disabled = True)]
        input_X[column_names[8]] = [st_columns[8].text_input(f'{column_names[8]}', value = reported_date.day, disabled = True)]
        input_X[column_names[9]] = [st_columns[9].text_input(f'{column_names[9]}', value = reported_date.timetuple().tm_yday, disabled = True)]
        input_X[column_names[10]] = [st_columns[10].text_input(f'{column_names[10]}', value = reported_date.strftime("%A"), disabled = True)]
        input_X[column_names[11]] = [st_columns[11].time_input(f'{column_names[11]}', value = datetime.time(int(mode_data[column_names[11]]), 0, 0)).hour]


        input_X[column_names[12]] = [st_columns[12].text_input(f'{column_names[12]}', value = occurence_date.year, disabled = True)]
        input_X[column_names[13]] = [st_columns[13].text_input(f'{column_names[13]}', value = occurence_date.strftime("%B"), disabled = True)]
        input_X[column_names[14]] = [st_columns[14].text_input(f'{column_names[14]}', value = occurence_date.day, disabled = True)]
        input_X[column_names[15]] = [st_columns[15].text_input(f'{column_names[15]}', value = occurence_date.timetuple().tm_yday, disabled = True)]
        input_X[column_names[16]] = [st_columns[16].text_input(f'{column_names[16]}', value = occurence_date.strftime("%A"), disabled = True)]
        input_X[column_names[17]] = [st_columns[17].time_input(f'{column_names[17]}', value = datetime.time(int(mode_data[column_names[17]]), 0, 0)).hour]
        
        input_X[column_names[18]] = [st_columns[19].selectbox(f'{column_names[18]}', ['NO', 'YES'])]
        input_X[column_names[19]] = [st_columns[20].selectbox(f'{column_names[19]}', neighbourhood)]
        input_X[column_names[20]] = [float(st_columns[21].text_input(f'{column_names[20]}', value = mode_data[column_names[20]]))]
        input_X[column_names[21]] = [float(st_columns[22].text_input(f'{column_names[21]}', value = mode_data[column_names[21]]))]
        
        X = pd.DataFrame(input_X)
        encoders = pickle.load(open('saved_models/LabelEncoder', 'rb'))
        for col in object_cols:
            X[col] = encoders[col].transform(X[col])
        
        predict = st.button('Make Prediction >>>')
        if predict:
            X = scale_data(X)
            pred = get_label(model.predict(X)[0])


            st.success(f'The Predicted label is: {pred}')

        
        