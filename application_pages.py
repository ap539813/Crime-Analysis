import streamlit as st
from assets import theme_image_name, logistic_regression, decision_tree, standerdscaler
from prediction import make_prediction
from visualization import show

def main():
    st.sidebar.image(theme_image_name)

    st.sidebar.title("Control Panel")

    type_model = st.sidebar.radio("Select Type of Model: ", ('Data Visualization', 'Logistic Regression', 'Model 2'))

    if type_model == 'Data Visualization':
        st.title(f"Data Visualization")
        show()

    elif type_model == 'Logistic Regression':
        st.title(f"Major Crime Classification Using {type_model}")

        make_prediction(logistic_regression)

    

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


