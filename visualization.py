import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

data = pd.read_csv('Data/Major_Crime_Indicators.csv')
data.drop(['event_unique_id', 'Hood_ID', 'X', 'Y', 'Index_'], axis = 1, inplace = True)
data = data[data['Longitude'] != 0]

def show():

    show_data = st.checkbox('Show data', key = 1)
    if show_data:
        st.markdown('## Major Crime Indicator raw data')
        st.dataframe(data)


    
    c1, c2 = st.columns([4, 1])
    c1.markdown('## Count of the incident with respec to year')
    hide_plot1 = c1.checkbox('Hide graph', key = 2)
    if not hide_plot1:
        show_count_per_crime = c2.checkbox('Include Major Crime types', key = 3)
        if show_count_per_crime:
            fig = plt.figure(figsize = (20, 6))
            sns.countplot(x = 'occurrenceyear', data = data, hue = 'mci_category')
            plt.xticks(rotation = 90)
            st.pyplot(fig)

        else:
            fig = plt.figure(figsize = (20, 6))
            sns.countplot(x = 'occurrenceyear', data = data)
            plt.xticks(rotation = 90)
            st.pyplot(fig)

    c3, c4 = st.columns([4, 1])
    c3.markdown('## The variation of the number of crimes per month')
    hide_plot2 = c3.checkbox('Hide graph', key = 4)
    if not hide_plot2:
        show_count_per_crime = c4.checkbox('Include Major Crime types', key = 5)
        if show_count_per_crime:
            months = ['January', 'February', 'March',
                        'April', 'May', 'June', 
                        'July', 'August', 'September', 
                        'October', 'November', 'December']

            incedent_by_months = data.groupby(['occurrencemonth', 'mci_category']).count().loc[months, 'Division'].reset_index()

            fig = plt.figure(figsize = (20, 6))
            sns.barplot(x = incedent_by_months.values[:, 0], y = incedent_by_months.values[:, 2], hue = incedent_by_months.values[:, 1])
            plt.xticks(rotation = 90)
            st.pyplot(fig)

        else:
            months = ['January', 'February', 'March',
                        'April', 'May', 'June', 
                        'July', 'August', 'September', 
                        'October', 'November', 'December']

            incedent_by_months = data.groupby(['occurrencemonth']).count().loc[months, 'Division'].reset_index()

            fig = plt.figure(figsize = (20, 6))
            sns.barplot(x = incedent_by_months.values[:, 0], y = incedent_by_months.values[:, 1])
            plt.xticks(rotation = 90)
            st.pyplot(fig)


    c4, c5 = st.columns([4, 1])
    c4.markdown('## The plot of no of incidents by each year')
    hide_plot3 = c4.checkbox('Hide graph', key = 6)
    if not hide_plot3:
        yearly_distribution = c5.checkbox('Yearly distribution', key = 7)
        if not yearly_distribution:
            fig = plt.figure(figsize = (20, 6))
            # It is necessary to understand the areas whichare affected by the crime in the city

            loc_count = data[['Longitude','Latitude']].value_counts()
            Longitude = np.array([a for a, _ in loc_count.index])
            Latitude = np.array([b for _, b in loc_count.index])
            count_crimes = loc_count.values

            fig = px.density_mapbox(lat=Latitude, lon=Longitude, z=count_crimes, radius=10,
                                    center=dict(lat=43.717, lon=-79.39906), zoom=9.5, width=800, height=500,
                                    mapbox_style="stamen-terrain",
                                    title="Number of incidents in each area")
            st.plotly_chart(fig, use_container_width=True)

        else:
            # the plot of no of incidents by each year
            year = st.slider('Select Year', min_value = 2014, max_value = 2022,)
            loc_count = data[data['occurrenceyear'] == year][['Longitude','Latitude']].value_counts()
            Longitude = np.array([a for a, _ in loc_count.index])
            Latitude = np.array([b for _, b in loc_count.index])
            count_crimes = loc_count.values

            fig = px.density_mapbox(lat = Latitude, lon = Longitude, z = count_crimes, radius = 10,
                                    center = dict(lat = 43.717, lon = -79.39906), zoom = 9.5, width = 800, height = 500,
                                    mapbox_style = "stamen-terrain",
                                    title = f"Number of incidents in each area in year {year}")
            st.plotly_chart(fig, use_container_width=True)

