import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset (assuming it is in the same directory)
def load_data():
    return pd.read_csv("covid_19_clean_complete.csv")

data = load_data()

# Set Streamlit page configuration
st.set_page_config(page_title="Data Analysis Presentation", layout="wide")


# Add the header and group name
st.title("Data Analysis Techniques")
st.subheader("Group Name: Third Streak")

# Tabs for Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Exploration", "Analysis and Insights", "Conclusions and Recommendations"])

# Tab 1: Overview
with tab1:
    st.title("Overview")
    st.write("""
    This app explores the findings from analyzing a COVID-19 dataset using Linear Regression to study the relationship between confirmed cases and deaths globally.
    """)

    st.subheader("Introduction")
    st.write("""
    The provided COVID-19 dataset includes detailed information about the spread and impact of COVID-19 across different regions worldwide.  
    This analysis aims to extract insights using Linear Regression to understand trends, patterns, and relationships in the data.  
    Linear Regression is used to identify relationships between cumulative case counts and deaths.
    """)

    st.subheader("Dataset Description")
    st.write("""
    The dataset consists of the following columns:
    1. **Province/State**: The specific administrative division within a country.  
    2. **Country/Region**: The country to which the data belongs.  
    3. **Lat**: Latitude of the location.  
    4. **Long**: Longitude of the location.  
    5. **Date**: Date of the cumulative report.  
    6. **Confirmed**: Cumulative confirmed cases until the specific date.  
    7. **Deaths**: Cumulative deaths until the specific date.  
    8. **Recovered**: Cumulative recovered cases until the specific date.  
    9. **Active**: Number of active cases on the specific date.  
    10. **WHO Region**: The World Health Organization region of the country.  
    """)

    st.write("Dataset link: [Kaggle - COVID-19 Dataset](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)")

    st.subheader("Objective")
    st.write("""
    We will use Linear Regression:
    - To study the relationship between **Confirmed Cases** and **Deaths**.  
    - This analysis helps in understanding how mortality correlates with confirmed cases globally.
    """)


# Tab 2: Data Exploration
with tab2:
    st.title("Data Exploration")
    st.write("""
    This section highlights key data preparation steps and exploratory findings.
    """)

    st.subheader("Random Sample")
    st.write("Here is a random sample of 5 records from the dataset:")
    st.write(data.sample(5))  # Display a random sample of 5 records

    # Count all rows in the dataset
    num_records = data.shape[0]
    st.write(f"**Number of records:** {num_records}")

    # Check for negative values in the 'Active' column
    negative_active = data[data['Active'] < 0]
    if negative_active.empty:
        st.write("No negative values found in the **Active** column.")
    else:
        st.write("Rows with negative values in the **Active** column:")
        st.write(negative_active)

    st.subheader("Basic Descriptive Statistics")
    st.write("Below are the descriptive statistics for the dataset:")

    # Display descriptive statistics
    st.write(data.describe())

    # Add observation
    st.subheader("Observations")
    st.write("""
    - The dataset contains **49,068 records**.
    - The average number of **confirmed cases** is **16,885**, while **deaths** average at **884**, 
        showing a case fatality ratio of approximately **5.2%**.
    - The highest number of **confirmed cases** is **4,290,259**, and the **median** is **168**.
    - **Active cases** show high variability, ranging from **0** to **2,816,444**.
    """)

    st.subheader("Correlation Matrix")
    st.write("Explore the relationships between key numeric fields in the dataset.")

    # Filter relevant fields for heatmap
    numeric_fields = ['Confirmed', 'Deaths', 'Recovered', 'Active']
    correlation_matrix = data[numeric_fields].corr()

    # Correlation heatmap using matplotlib and seaborn
    plt.figure(figsize=(6, 4))  # Smaller size
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    st.pyplot(plt)

    # Add observation
    st.subheader("Observations")
    st.write("""
    - The correlation matrix shows strong positive correlations between most variables.
    - **Confirmed cases** strongly correlate with **Active cases** (0.95), **Deaths** (0.91), and **Recovered** (0.90).
    - **Deaths** moderately correlate with **Recovered** (0.76) and strongly with **Active** (0.89).
    - **Recovered** and **Active cases** have the weakest, yet positive, correlation (0.71).
    - Overall, **confirmed cases drive trends** in all other variables.
    """)

# Tab 3: Analysis and Insights
with tab3:
    st.title("Analysis and Insights")
    st.write("""
    This section presents the key findings and visualizations derived from the analysis.
    Select an option from the dropdown to explore different features.
    """)
    
    # Dropdown for feature selection
    feature = st.selectbox(
        "Select Feature for Exploration",
        ["Scatter Plot: Confirmed Cases vs Deaths", "Regression Analysis: Confirmed Cases vs Deaths"]
    )


    if feature == "Scatter Plot: Confirmed Cases vs Deaths":
        st.subheader("Scatter Plot: Confirmed Cases vs Deaths (Log Scale)")
        st.write("""
        This scatter plot visualizes the relationship between confirmed cases and deaths, 
        with outliers removed for clearer insights. The color gradient represents the log scale of confirmed cases.
        """)

        # Filter data
        data = data[(data['Confirmed'] > 0) & (data['Deaths'] > 0)]
    
        # Remove outliers using quantile-based filtering
        lower_confirmed_threshold = data['Confirmed'].quantile(0.10)
        lower_deaths_threshold = data['Deaths'].quantile(0.70)

        filtered_data = data[
            (data['Confirmed'] >= lower_confirmed_threshold) &
            (data['Deaths'] >= lower_deaths_threshold)
        ]

        # Scatter plot with log scale
        plt.figure(figsize=(6, 4))  # Smaller size
        colors = np.log10(filtered_data['Confirmed'])

        scatter = plt.scatter(
            filtered_data['Confirmed'],
            filtered_data['Deaths'],
            c=colors,
            cmap='viridis',  # Color map
            s=10,            # Marker size
            alpha=0.7        # Transparency
        )

        # Add a color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Log10(Confirmed Cases)', fontsize=12)

        # Apply log scale to axes
        plt.xscale('log')
        plt.yscale('log')

        # Add titles, labels, and grid
        plt.title('Scatter Plot: Confirmed Cases vs Deaths (Log Scale, Outliers Removed)', fontsize=16)
        plt.xlabel('Confirmed Cases (log scale)', fontsize=12)
        plt.ylabel('Deaths (log scale)', fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Annotate an outlier
        max_confirmed = filtered_data['Confirmed'].max()
        max_deaths = filtered_data.loc[filtered_data['Confirmed'] == max_confirmed, 'Deaths'].values[0]
        plt.annotate(
            'Max Cases',
            xy=(max_confirmed, max_deaths),
            xytext=(10, 20),
            textcoords='offset points',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            fontsize=10
        )

        # Display the plot
        st.pyplot(plt)

        # Observations
        st.subheader("Observations")
        st.write("""
        - Regions with steep slopes may need to focus on improving healthcare systems or early detection.
        - Areas with flat slopes could be benefiting from better healthcare interventions or younger populations.
        - The scatter plot offers a valuable tool for understanding regional differences, healthcare effectiveness, and pandemic stages, guiding targeted responses and interventions.
        """)

    elif feature == "Regression Analysis: Confirmed Cases vs Deaths":
        st.subheader("Regression Analysis: Confirmed Cases vs Deaths")
        st.write("""
        This feature performs a linear regression analysis on log-transformed data to study the relationship 
        between Confirmed Cases and Deaths. The regression line is overlaid on the scatter plot for visualization.
        """)

        # Filter data
        data = data[(data['Confirmed'] > 0) & (data['Deaths'] > 0)]
    
        # Remove outliers using quantile-based filtering
        lower_confirmed_threshold = data['Confirmed'].quantile(0.10)
        lower_deaths_threshold = data['Deaths'].quantile(0.70)

        filtered_data = data[
            (data['Confirmed'] >= lower_confirmed_threshold) &
            (data['Deaths'] >= lower_deaths_threshold)
        ]

        # Prepare data for regression
        X = filtered_data['Confirmed'].to_numpy().reshape(-1, 1)  # Independent variable (Confirmed cases)
        y = filtered_data['Deaths'].to_numpy()                   # Dependent variable (Deaths)

        
        # Transform data for log-log regression
        log_X = np.log10(X)
        log_y = np.log10(y)

        # Initialize and fit the linear regression model
        model = LinearRegression()
        model.fit(log_X, log_y)

        # Extract slope and intercept
        slope = model.coef_[0]
        intercept = model.intercept_

        # Display regression equation
        st.write(f"### Regression Equation: log10(Deaths) = {slope:.4f} * log10(Confirmed) + {intercept:.4f}")

        # Generate predictions for the regression line
        log_X_range = np.linspace(log_X.min(), log_X.max(), 500).reshape(-1, 1)
        log_y_pred = model.predict(log_X_range)

        colors = np.log10(filtered_data['Confirmed'])

        # Scatter plot with regression line
        plt.figure(figsize=(6, 4))  # Smaller size
        scatter = plt.scatter(
            filtered_data['Confirmed'],
            filtered_data['Deaths'],
            c=colors,
            cmap='viridis',
            s=10,
            alpha=0.7,
            label='Data'
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('Log10(Confirmed Cases)', fontsize=12)

        # Plot the regression line
        plt.plot(
            10 ** log_X_range.flatten(),
            10 ** log_y_pred,
            color='red',
            linewidth=2,
            label=f'Regression Line: slope={slope:.2f}'
        )

        # Apply log scale to axes
        plt.xscale('log')
        plt.yscale('log')

        # Add titles, labels, grid, and legend
        plt.title('Scatter Plot with Linear Regression: Confirmed Cases vs Deaths', fontsize=16)
        plt.xlabel('Confirmed Cases (log scale)', fontsize=12)
        plt.ylabel('Deaths (log scale)', fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=12)

        # Display the plot
        st.pyplot(plt)

        # Observations
        st.subheader("Observations")
        st.write("""
        - The regression analysis shows a clear, predictable relationship between Confirmed Cases and Deaths.
        - The sub-linear relationship suggests that deaths increase more slowly as cases rise.
        - Healthcare preparedness, widespread testing, and targeted interventions are crucial for minimizing fatalities.
        - This model can help policymakers and healthcare systems allocate resources effectively to reduce the impact of future outbreaks.
        """)

# Tab 4: Conclusions and Recommendations
with tab4:
    st.title("Conclusions and Recommendations")
    st.write("""
    This section provides a summary of the analysis and actionable recommendations based on the findings.
    """)
    
    st.subheader("Main Takeaways")
    
    st.markdown("""
    **1. Log-Log Power-Law Relationship**  
    - **Observation**: A strong power-law relationship exists between Confirmed Cases and Deaths, appearing linear on a log-log scale.  
    - **Insight**: The slope of the regression line indicates a sub-linear relationship, suggesting that deaths increase at a slower rate than confirmed cases as the pandemic progresses. This implies that the case fatality ratio may decline as case numbers rise.  
    """)
    
    st.markdown("""
    **2. Impact of Healthcare and Interventions**  
    - **Sub-linear Trend**: The slope (less than 1) points to adaptive responses in healthcare systems and the efficacy of early interventions in reducing fatalities.  
    - **Broader Testing**: Increased testing and reporting of less severe or asymptomatic cases likely contribute to a lower observed fatality rate over time.  
    """)
    
    st.markdown("""
    **3. Outlier Removal and Data Quality**  
    - **Outlier Mitigation**: Removing extreme values ensures that the analysis reflects broader trends without being skewed by anomalies.  
    - **Data Variability**: The findings remain sensitive to regional variations in healthcare infrastructure, data reporting accuracy, and policy measures.  
    """)
    
    st.markdown("""
    **4. Predictive Power**  
    - **Estimation Tool**: The regression model can predict deaths based on confirmed cases, aiding in healthcare resource allocation, such as ICU beds and ventilators.  
    - **Policy Planning**: These insights help inform targeted strategies to mitigate the pandemic's impact and prepare for future outbreaks.  
    """)
    
    st.subheader("Recommendations")
    st.markdown("""
    - **Strengthen Healthcare Systems**: Invest in scalable healthcare resources to accommodate surges in cases and minimize fatalities.  
    - **Enhance Testing**: Increase testing to identify mild or asymptomatic cases early, reducing the overall fatality rate.  
    - **Focus on Regional Analysis**: Conduct detailed regional studies to account for variations in healthcare capacity and policy effectiveness.  
    - **Improve Data Reporting**: Encourage standardized and transparent reporting of pandemic data to enhance the reliability of predictive models.  
    - **Leverage Predictive Insights**: Use regression analysis outcomes to guide resource allocation and policy decisions for more effective pandemic management.  
    """)
