import streamlit as st
from .exploratory import plot_yearly_temp, plot_yearly_precip, plot_avgtemp_bydistrict, plot_boxplot, plot_avgprecip_bydistrict,plot_by_seasons

def show_analysis(df):
    """
    Display exploratory data analysis
    """
    st.header("Exploratory Data Analysis")

    # show raw data
    st.subheader("Raw Data")
    st.dataframe(df.sample(10))

    # basic statistics
    st.subheader("Descriptive Analysis") # Statistical summary
    st.write(df["Temp_2m"].describe())

    # Plot the time series
    st.subheader("Yearly temperature of nepal")
    fig = plot_yearly_temp(df)
    st.pyplot(fig)

    # Plot Average Temperature by disctrict
    st.subheader("Yearly precipitation of nepal")
    fig = plot_yearly_precip(df)
    st.pyplot(fig)

    # Plot Average Temperature by disctrict
    st.subheader("Average Temperature by disctrict")
    fig = plot_avgtemp_bydistrict(df)
    st.pyplot(fig)

    # Plot Average Precipitation by disctrict
    st.subheader("Average Precipitation by disctrict")
    fig = plot_avgprecip_bydistrict(df)
    st.pyplot(fig)

    # Season wise Voilin plot
    st.subheader("Voilin plot by seasons")
    fig = plot_by_seasons(df)
    st.pyplot(fig)

    # Box plot
    st.subheader("Box plot for outliers")
    fig = plot_boxplot(df)
    st.pyplot(fig)

    

    


