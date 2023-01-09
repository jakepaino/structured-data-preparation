import streamlit.sync as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Structured Data Preparation")
# Allow the user to upload a file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display basic information about the DataFrame
    st.info("First 5 Rows of Data: Does everything look correct?")
    st.write(df.head())
    st.info("Data Types: Will you need to reformat any of these to numeric or datetime?")
    st.write(df.dtypes)
    st.info("Number of missing data per column: If you have missing values, How are you going to handle them?")
    st.write(df.isna().sum())
    st.info("Statistical Summary of Numeric data.")
    st.write(df.describe())

    st.info("Take a look at the Correlation Matrix and Histogram plots of numeric data."
            " Look for colinearity and outliers, you may want to drop a column or row.")
    # Allow the user to choose which plots to display
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plot_type = st.radio("Select a plot type",
                         ("Correlation matrix", "Histograms"))

    if plot_type == "Correlation matrix":
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display a correlation matrix using a heatmap
        ax = sns.heatmap(df.corr(), annot=True, fmt=".2f")
        st.pyplot(fig)
    elif plot_type == "Histograms":
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(18, 18))

        # Display histograms for all numeric columns
        ax = df.hist()
        plt.tight_layout()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

    # Ask the user if they want to remove any outliers
    remove_outliers = st.radio(
        "Does the data have any outliers you would like to remove?", ("Yes", "No"))

    # Check if the user wants to remove outliers
    if remove_outliers == "Yes":
        # Get a list of the columns in the DataFrame
        columns = df.columns

        # Display a multiple select box to allow the user to select one or more columns to check for outliers
        columns_to_check = st.multiselect(
            "Select one or more columns with outliers you would like to remove", columns)

        # Display a dropdown menu to allow the user to select a comparison operator
        operator = st.selectbox(
            "Select a comparison operator for the data you want removed", (">", ">=", "<", "<="))

        # Ask the user to enter a value to use as the cutoff
        cutoff = st.number_input("Enter a value to use as the cutoff")

        # Iterate through the selected columns
        for col in columns_to_check:
            # Remove the outliers using the selected operator and cutoff
            if operator == ">":
                df = df[df[col] <= cutoff]
            elif operator == ">=":
                df = df[df[col] < cutoff]
            elif operator == "<":
                df = df[df[col] >= cutoff]
            elif operator == "<=":
                df = df[df[col] > cutoff]
    else:
        st.info("You have selected that you have no outliers you would like to remove.")

    non_numeric_columns = df.select_dtypes(
        exclude=["float", "int", "datetime"]).columns
    if len(non_numeric_columns) > 0:
        # Display a message about converting dtypes
        st.info("Now let's change all of the columns into numeric type or datetime."
                " Select the appropriate strategy for the given column."
                " If you would like to drop the column now, you can select drop column.")
        # Iterate through the non-numeric columns
        for col in non_numeric_columns:
            # Display a dropdown menu to allow the user to select a conversion strategy
            strategy = st.radio(f"Select a strategy for column '{col}'", (
                "get_dummies", "to_numeric", "mapping dictionary", "to_datetime", "drop column"))

            if strategy == "to_datetime":
                # Convert the column to datetime using pandas.to_datetime
                df[col] = pd.to_datetime(df[col])
            elif strategy == "get_dummies":
                # Create dummy variables for the column using pandas.get_dummies and drop the first dummy column
                df = pd.concat([df.drop(col, axis=1), pd.get_dummies(
                    df[col], prefix=col, drop_first=True)], axis=1)
            elif strategy == "to_numeric":
                # Convert the column to numeric using pandas.to_numeric
                df[col] = pd.to_numeric(df[col])
            elif strategy == "mapping dictionary":
                # Create a mapping dictionary to convert the column to numeric
                categories = df[col].unique()
                mapping = {category: i for i,
                           category in enumerate(categories)}
                df[col] = df[col].map(mapping)
            elif strategy == "drop column":
                # Drop the column from dataframe
                df = df.drop(columns=[col])
    else:
        st.info("You have no non-numeric columns!!")

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        # Display a message indicating that missing values were found
        st.warning("**Missing values were found in the dataset**")

        # Get a list of columns with missing values
        missing_columns = df.columns[df.isnull().any()]

        # Iterate through the columns with missing values
        for col in missing_columns:
            # Display a dropdown menu to allow the user to select a filling strategy
            fill_strategy = st.radio(f"Select a strategy for filling missing values in column '{col}'", (
                "dropna", "fill with mean", "fill with median", "fill with specific number"))

            if fill_strategy == "dropna":
                # Drop the rows with missing values
                df = df.dropna(subset=[col])
            elif fill_strategy == "fill with mean":
                # Fill the missing values with the mean of the column
                df[col] = df[col].fillna(df[col].mean())
            elif fill_strategy == "fill with median":
                # Fill the missing values with the median of the column
                df[col] = df[col].fillna(df[col].median())
            elif fill_strategy == "fill with specific number":
                # Ask the user to enter a number to use to fill the missing values
                fill_value = st.number_input(
                    "Enter a number to use to fill the missing values")
                # Fill the missing values with the specified number
                df[col] = df[col].fillna(fill_value)
    else:
        # Display a message indicating that no missing values were found
        st.info("No missing values were found in the dataset")

    # Ask the user if they want to drop any columns
    drop_columns = st.radio(
        "Would you like to drop any columns?", ("Yes", "No"))

    # Check if the user wants to drop any columns
    if drop_columns == "Yes":
        # Get a list of the columns in the DataFrame
        columns = df.columns

        # Display a multiple select box to allow the user to select one or more columns to drop
        columns_to_drop = st.multiselect(
            "Select the column(s) you would like to drop", columns)

        # Drop the selected columns
        df = df.drop(columns_to_drop, axis=1)
    else:
        # Display the DataFrame
        st.info("You do not wish to drop any columns.")
    st.write(df)

    # Ask the user if they want to rename any columns
    rename_columns = st.radio(
        "Would you like to rename any columns?", ("Yes", "No"))

    # Check if the user wants to rename any columns
    if rename_columns == "Yes":
        # Display a multiselect widget to allow the user to select the columns to rename
        columns_to_rename = st.multiselect(
            "Select the columns to rename", df.columns)

        # Iterate through the columns to rename
        for col in columns_to_rename:
            # Display an input widget to allow the user to enter the new name
            new_name = st.text_input(f"Enter the new name for column '{col}'")

            # Rename the column
            df = df.rename(columns={col: new_name})
    else:
        # Display a message indicating that the data is almost model-ready
        st.info("You do not want to rename any columns")

    st.dataframe(df.head(0))

    st.info("Your data is almost model-ready!!!"
            "  If you know your dependent variable, select the box and we will move it to the first column.")

    # Ask the user if they know which variable is the dependent variable
    dependent_variable = st.radio(
        "Do you know which variable is the dependent variable?", ("No", "Yes"))

    if dependent_variable == "Yes":
        # Display a selectbox to allow the user to select the dependent variable
        dependent_var = st.selectbox(
            "Select the dependent variable", df.columns)

        # Move the dependent variable to the first column
        df = df[[dependent_var] +
                [col for col in df.columns if col != dependent_var]]
    else:
        pass
    st.dataframe(df.head())
    st.info("How is your data looking now?"
            "  If it looks good, you are ready to save, if not go back up and make the necesssary changes.")
    st.info("Any non-numeric columns?")
    st.write(df.dtypes)
    st.info("Any missing values?")
    st.write(df.isna().sum())

    # Display a message indicating that the data is ready to be saved
    st.info("We are now ready to save your model-ready data. Be sure to save it as a different name so it does not overwrite the original data.")

    # Ask the user to enter the path to the directory to save the file in
    save_directory = st.select_folder("Select a directory to save the file")
    
    # Ask the user to enter a file name
    file_name = st.text_input("Enter a file name for the saved data")

    # Save the cleaned DataFrame as a CSV file in the specified directory
    df.to_csv(f"{save_directory}/{file_name}.csv", index=False)
