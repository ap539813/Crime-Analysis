import streamlit as st
from assets import theme_image_name, model_path
from prediction import make_prediction
from visualization import show
import os

def main():
    st.sidebar.image(theme_image_name)

    st.sidebar.title("Control Panel")
    model_list = os.listdir(model_path)

    type_model = st.sidebar.radio("Select Type of Model: ", ['Data Visualization'] + model_list)

    if type_model == 'Data Visualization':
        st.title(f"Data Visualization")
        show()

    else:
        st.title(f"Major Crime Classification Using {type_model.split('(')[0]}")
        st.write('**Please fill the values of different attributes asked below, some of the field are disabled on purpose, the values in those columns will be filled auromatically**')

        make_prediction(model_path + type_model)


    

def homepage():
    home_image = st.image(theme_image_name)

    c1, c2, c3 = st.columns([2,1,2])


    c2.markdown('')
    c2.markdown('')
    continue_forward = c2.button('Continue >>>')

    st.session_state['home_page'] = False    

    if continue_forward:
        print('going to the application!!')
        home_image.empty()
        main()


