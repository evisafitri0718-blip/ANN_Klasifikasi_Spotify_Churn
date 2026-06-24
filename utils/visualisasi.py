

import streamlit as st
import pandas as pd
import plotly.express as px

def churn_chart(df):

    fig = px.pie(
        df,
        names='is_churned',
        title='Distribusi Churn'
    )

    st.plotly_chart(fig, use_container_width=True)


def gender_chart(df):

    fig = px.histogram(
        df,
        x='gender',
        color='is_churned',
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)