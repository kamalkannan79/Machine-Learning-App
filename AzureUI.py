#Loading neccesary packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.linear_model as glm
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.preprocessing as pre
import plotly_express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from PIL import Image

img1 = Image.open("ML_app_1.jpeg")
st.set_page_config(page_title='The Machine Learning App',
    page_icon=img1,
    layout="centered",
    initial_sidebar_state="auto")
# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# title of the app
st.title("The Machine Learning App")


# Add a sidebar
st.sidebar.subheader("The Machine Learning Settings")

# Setup file upload
uploaded_file = st.sidebar.file_uploader(
                        label="Upload your CSV or Excel file",
                         type=['csv', 'xlsx'])

global df

if uploaded_file is not None:
    print(uploaded_file)
    print("hello")

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)
 
st.sidebar.write("(or)")
st.sidebar.write("Choose some default datasets")

default_dataset = st.sidebar.selectbox(
    label = "select the dataset",
    options=['None', 'Campus Requirement Dataset', 'Carseat Dataset', 'Cereal Dataset', 'Insurance Dataset', 'Iris Dataset', 'Mtcars Dataset', 'Penguin Dataset', 'Pokemon Dataset','Students Dataset','Students Test Performance Dataset']
)

if default_dataset == 'Campus Requirement Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/Campus Requirement.csv")
    except Exception as e:
        print(e)   

if default_dataset == 'Carseat Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/carseats.csv")
    except Exception as e:
        print(e)  

if default_dataset == 'Cereal Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/cereal.csv")
    except Exception as e:
        print(e)  

if default_dataset == 'Insurance Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/insurance.csv")
    except Exception as e:
        print(e)  

if default_dataset == 'Iris Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/IRIS.csv")
    except Exception as e:
        print(e)         

if default_dataset == 'Mtcars Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/mtcars.csv")
    except Exception as e:
        print(e)

if default_dataset == 'Penguin Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/Penguins_data.csv")
    except Exception as e:
        print(e)

if default_dataset == 'Pokemon Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/Pokemon.csv")
    except Exception as e:
        print(e)

if default_dataset == 'Students Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/students.csv")
    except Exception as e:
        print(e)
        
if default_dataset == 'Students Test Performance Dataset':
    try:
        df = pd.read_csv("C:/Users/kamalesh/Desktop/Project Dataset/StudentsPerformance.csv")
    except Exception as e:
        print(e)


st.subheader("1.Data")
global numeric_columns
global non_numeric_columns
try:
    st.write(df)
    numeric_columns = list(df.select_dtypes(['float','int','float32','int32','float64','int64']).columns)
    non_numeric_columns = list(df.select_dtypes(['object','bool']).columns)
    non_numeric_columns.append(None)
    print(non_numeric_columns)
except Exception as e:
    print(e)
    st.write("Please upload file to the application.")
    
content = st.sidebar.radio(
    label="Select the content",
    options=['Home','Data Preprocessing','Charts','Algorithm']
)

st.header((content))
if content == "Home":
    try:
        st.write("Machine Learning algorithm using applications")
        st.image(img1,width = 800)
       
        
    except Exception as e:
            print(e)
            
elif content == "Data Preprocessing":
    content_DP = st.sidebar.selectbox(
    label = "select the data preprocess",
    options = ["EDA","Data Cleaning"])
    if content_DP == "EDA":
        try:
            st.write("1.It shows the First 5 rows of data")
            head_df = df.head()
            st.write(head_df)
            ############
            st.write("2.It shows the last 5 rows of data")
            tail_df = df.tail()
            st.write(tail_df)
            ########
            st.write("3.It shows the dimension of the data")
            shape_df = df.shape
            st.write(shape_df)
            ################
            st.write("4.It shows the description of the data")
            describe_df = df.describe()
            st.write(describe_df)
            ###########
            st.write("5.Identify the Duplicates")
            st.write("There Exist {:,} duplications in our data,".format(df.duplicated().sum()))
            st.write("In this data we have some duplicate values.so we have to remove the duplicate data")
            duplicate = df.drop_duplicates(inplace=True)
            st.code("duplicate = df.drop_duplicates(inplace=True)")
            st.write("We check the Dimension of the data")
            head_shape = df.shape
            st.write(head_shape)
            #############
            
        except Exception as e:
                print(e)
                
    if content_DP == "Data Cleaning":
        numeric = list(df.select_dtypes(['float','int','float32','int32','float64','int64']).columns)
        non_numeric = list(df.select_dtypes(['object','bool']).columns)
        type_ = st.sidebar.radio(
        label = "select the types",
        options = ["Mean","Median","Mode"])
        st.subheader("Null value Handling")
        st.write("First we have to check the null value in the dataset")
        st.write("If its shows True then the data contains a null values we have to clean that noise data.")
        st.write("If its shows False then there is no null values in the data.")
        null_df = df.isnull().values.any()
        st.write(null_df)
        st.write("It comes True so we have to clean the null value")
        

        ##########
        st.write("Null Values Percentage")
        st.write("It shows the null values percentage of each columns in the data")
        null_percent_df = df.isnull().sum()*100/df.shape[0]
        st.write(null_percent_df)
        if type_ == "Mean":
            numeric = st.sidebar.selectbox("numeric_columns",numeric)
            try:
                df[numeric] = df[numeric].fillna(value = df[numeric].mean())
                st.write("After Data Cleaning Process,It shows the null values in the dataset.")
                st.write(df.isnull().sum())
            except Exception as e:
                    print(e)
                    
        if type_ == "Median":
            numeric = st.sidebar.selectbox("numeric_columns",numeric)
            try:
                df[numeric] = df[numeric].fillna(df[numeric].astype(float).median(skipna = True),inplace = True)
                st.write("After Data Cleaning Process,It shows the null values in the dataset.")
                st.write(df.isnull().sum())
            except Exception as e:
                    print(e)
                    
        if type_ == "Mode":
            non_numeric = st.sidebar.selectbox("non_numeric_columns",non_numeric)
            try:
                df[non_numeric] = df[non_numeric].fillna(df[non_numeric].mode()[0])
                st.write("After Data Cleaning Process,It shows the null values in the dataset.")
                st.write(df.isnull().sum())
            except Exception as e:
                    print(e)

elif content == 'Charts':
    chart_select = st.sidebar.selectbox(
    label="Select the chart type",
    options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot', 'Heatmap', 'Barchart'])
    df_new = df.dropna()
    if chart_select == "Scatterplots":
        st.sidebar.subheader("Scatterplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.scatter(data_frame=df_new, x=x_values, y=y_values, color=color_value)
            # display the chart
            st.plotly_chart(plot)
            st.header("Inference")
            st.write("Generally Scatterplot helps to find the relation between the variables. The x axis and Y axis Of the graph is ",x_values," and ",y_values," respectively. So we can find the graph shows the releationship between ", x_values, "and", y_values, ". Each color of the graph represents the ",color_value,"'s.")
        except Exception as e:
            print(e)
            

    if chart_select == 'Lineplots':
        st.sidebar.subheader("Line Plot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.line(data_frame=df_new, x=x_values, y=y_values, color=color_value)
            st.plotly_chart(plot)
            st.header("Inference")
            st.write("A lineplot is a graph that displays data as points or check marks above a number line, showing the frequency of each value. In the above graph we take  x axis and Y axis as is ",x_values," and ",y_values,". Each color of the graph represents the ",color_value,"'s. It shows the difference between each ",color_value,".")
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        try:
            x = st.sidebar.selectbox('Feature', options=numeric_columns)
            bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                         max_value=100, value=40)
            plot = px.histogram(x=x, data_frame=df_new,nbins = bin_size )
            st.plotly_chart(plot)
            st.header("Inference")
            st.write("A Histogram is a graphical representation of a grouped frequency distribution with continuous classes. Histogram is a univariate graph. In this graph we plotted a histogram for ",x,". In this we can find the frequency whether it is maximum frequency or minimum frequency in the ",x,". ")
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        try:
            y = st.sidebar.selectbox("Y axis", options=numeric_columns)
            x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.box(data_frame=df_new, y=y, x=x, color=color_value)
            st.plotly_chart(plot)
            st.header("Inference")
            st.write("This Graph Is Box Plot. Which shows us the lower limit(min),quartile 1,quartile 2,quartile 3 and upper limit(max). Which helps to find how the data is spreaded.In this graph we can see that maximum data is spread between ",x," and ",y,".")
        except Exception as e:
            print(e)
            
    if chart_select == 'Heatmap':
        st.sidebar.subheader("Heatmap Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.density_heatmap(data_frame=df_new, x=x_values, y=y_values)
            st.plotly_chart(plot)
            st.header("Inference")
            st.write("A heat map  is a data visualization technique that shows magnitude of a phenomenon as color in two dimensions. In this graph we taken X and Y axis as ",x_values," and ",y_values,". On the right side of the graph we can saw a count which has some color code for the values. We can understand the graph through the colors in it.")
        except Exception as e:
            print(e)

    if chart_select == 'Barchart':
        st.sidebar.subheader("Bar Chart Settings")
        try:
            y_values = st.sidebar.selectbox("Y axis", options=numeric_columns)
            x_values = st.sidebar.selectbox("X axis", options=non_numeric_columns)
            plot = px.bar(data_frame=df_new, y=y_values, x=x_values)
            st.plotly_chart(plot)
            st.header("Inference")
            st.write("Generally bar graph is helps to find the difference between categorical data. In this graph we can easily find the difference between ",x_values," about their ",y_values,". It helps easily  to understand that which ",x_values," have more ",y_values," and which ",x_values," have less ",y_values,".")
        except Exception as e:
            print(e)
            
elif content == "Algorithm":
    Algorithm_select = st.sidebar.selectbox(
    label = "Select the Algorithm",
    options = ["Linear Regression","LogisticRegression","RandomForestRegressor","Decision Tree","Neural Network"])
    df_new = df.dropna()
    st.subheader("Split Data")
    le = pre.LabelEncoder()
    for x in df_new.select_dtypes(include = 'object').columns.tolist():
        print(x)
        df_new[x]= le.fit_transform(df_new[x])
    X = df_new.iloc[:,:-1] # Using all column except for the last column as X
    Y = df_new.iloc[:,-1] # Selecting the last column as Y
    # Data splitting
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100,random_state = 12)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)
    
    if Algorithm_select == "Linear Regression":
        try:
            with st.sidebar.header("3.Learning Parameters"):
                parameter_fit_intercept = st.sidebar.select_slider("fit_intercept:Whether to calculate the intercept for this model.",options=[True, False])
                parameter_normalize = st.sidebar.select_slider('normalize:This parameter is ignored when fit_intercept is set to False.',options=[False, True])
                parameter_copy_X = st.sidebar.select_slider("copy_X:If True, X will be copied; else, it may be overwritten.",options=[True, False])
                parameter_positive= st.sidebar.select_slider("positive:When set to True, forces the coefficients to be positive.",options=[False, True])
                parameter_n_jobs = st.sidebar.slider("n_jobs:The number of jobs to use for the computation.",-1,0,1)
                
            lm = linear_model.LinearRegression(
               
                fit_intercept=parameter_fit_intercept,
                normalize=parameter_normalize,
                copy_X=parameter_copy_X,
                n_jobs=parameter_n_jobs,
                positive=parameter_positive
            )
            lm.fit(X_train, Y_train)
            
            st.subheader('2. Model Performance')

            st.markdown('**2.1. Training set**')
            Y_pred_train = lm.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_train, Y_pred_train) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )

            st.markdown('**2.2. Test set**')
            Y_pred_test = lm.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

            st.subheader('3. Model Parameters')
            st.write(lm.get_params()) 
            
            st.subheader("4.Model Score")
            st.write("Return the mean accuracy on the given test data and labels.")
            st.write(lm.score(X_test,Y_test))
            
            st.subheader("5.Model coefficients")
            st.write(lm.coef_)
            
            
        except Exception as e:
            print(e)
    
    if Algorithm_select == "LogisticRegression":
        try:           
            with st.sidebar.header("3.Learning Parameters"):
                parameter_penalty = st.sidebar.select_slider("penalty:Used to specify the norm used in the penalization.",options = ['l2','l1'])
                parameter_random_state = st.sidebar.slider('random_state:Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data. ',1,12,21,121)
                parameter_solver = st.sidebar.select_slider("solver:Algorithm to use in the optimization problem.",options = ['lbfgs','liblinear','saga'])
                parameter_max_iter= st.sidebar.slider("max_iter:Maximum number of iterations taken for the solvers to converge.",2000,2500,3000)
                parameter_n_jobs = st.sidebar.slider("n_jobs:Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”.",-1,0,1)
                
            logistic_model = glm.LogisticRegression( penalty=parameter_penalty,
                random_state=parameter_random_state,
                solver=parameter_solver,
                max_iter=parameter_max_iter,
                n_jobs=parameter_n_jobs)
            logistic_model.fit(X_train, Y_train)
            
            st.subheader('2. Model Performance')

            st.markdown('**2.1. Training set**')
            Y_pred_train = logistic_model.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_train, Y_pred_train) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )

            st.markdown('**2.2. Test set**')
            Y_pred_test = logistic_model.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

            st.subheader('3. Model Parameters')
            st.write(logistic_model.get_params())
                
            st.subheader("4.Model Score")
            st.write("Return the mean accuracy on the given test data and labels.")
            st.write(logistic_model.score(X_test,Y_test))
        except Exception as e:
            print(e)
            
    if Algorithm_select == "RandomForestRegressor":
        try:           
            with st.sidebar.subheader('3.Learning Parameters'):
                parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
                parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
                parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
                parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
            with st.sidebar.subheader('2.2. General Parameters'):
                parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
                parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
                parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
                parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
                parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

                
            rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                random_state=parameter_random_state,
                max_features=parameter_max_features,
                criterion=parameter_criterion,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score,
                n_jobs=parameter_n_jobs)
            rf.fit(X_train, Y_train)
                
            st.subheader('2. Model Performance')

            st.markdown('**2.1. Training set**')
            Y_pred_train = rf.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_train, Y_pred_train) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )

            st.markdown('**2.2. Test set**')
            Y_pred_test = rf.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

            st.subheader('3. Model Parameters')
            st.write(rf.get_params())
            
            st.subheader('4.Model Score')
            st.write("Return the mean accuracy on the given test data and labels.")
            st.write(rf.score(X_test,Y_test))
        except Exception as e:
            print(e)
            
    if Algorithm_select == "Decision Tree":
        try:
            with st.sidebar.header("3.Learning Parameters"):
                parameter_criterion = st.sidebar.select_slider("criterion:The function to measure the quality of a split.",options = ['gini','entropy'])
                parameter_random_state = st.sidebar.slider('random_state:  Controls the randomness of the estimator.',1,12,21,121)
                parameter_splitter = st.sidebar.select_slider("splitter:The strategy used to choose the split at each node.",options = ["best","random"])
                parameter_min_samples_split= st.sidebar.slider("min_samples_split:The minimum number of samples required to split an internal node",2,3,4)
                parameter_min_samples_leaf = st.sidebar.slider("min_samples_leaf:The minimum number of samples required to be at a leaf node.",1,2,3)
                
              
            DT = DecisionTreeClassifier(
                criterion=parameter_criterion,
                splitter= parameter_splitter,                
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf ,                
                random_state=parameter_random_state,
            )
            DT.fit(X_train, Y_train)
            
            st.subheader('2. Model Performance')

            st.markdown('**2.1. Training set**')
            Y_pred_train = DT.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_train, Y_pred_train) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )

            st.markdown('**2.2. Test set**')
            Y_pred_test = DT.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

            st.subheader('3. Model Parameters')
            st.write(DT.get_params())
            
            st.subheader("4. Model Score")
            st.write("Return the mean accuracy on the given test data and labels.")
            st.write(DT.score(X_test,Y_test))
            
            st.subheader("5.Cross Validation Score")
            from sklearn.model_selection import cross_val_score
            st.write(cross_val_score(DT, X, Y, cv=10))
            
        except Exception as e:
            print(e)
    
    if Algorithm_select == "Neural Network":
        try:
            with st.sidebar.header("3.Learning Parameters"):
                parameter_hidden_layer_sizes = st.sidebar.select_slider("hidden_layer_sizes:The ith element represents the number of neurons in the ith hidden layer.",options = [(30,30,30),(60,60,60),(90,90,90)])
                parameter_activation = st.sidebar.select_slider('activation:Activation function for the hidden layer.',options = ['identity', 'logistic', 'tanh', 'relu'])
                parameter_solver = st.sidebar.select_slider("solver:The solver for weight optimization.",options = ['lbfgs', 'sgd', 'adam'])
                parameter_alpha= st.sidebar.slider("alpha:L2 penalty (regularization term) parameter.",0.0001,0.01,0.001)
                parameter_learning_rate = st.sidebar.select_slider("learning_rate:Learning rate schedule for weight updates.",options = ['constant', 'invscaling', 'adaptive'])
                parameter_max_iter= st.sidebar.slider("max_iter:Maximum number of iterations.",200,400,600)
                parameter_shuffle = st.sidebar.select_slider("shuffle:Whether to shuffle samples in each iteration.",options = [True, False])
                parameter_random_state = st.sidebar.slider('random_state:  Controls the randomness of the estimator.',1,12,21,121)
                
            Neural = MLPClassifier(
                hidden_layer_sizes=parameter_hidden_layer_sizes,
                activation=parameter_activation,
                solver=parameter_solver,
                alpha=parameter_alpha,
                learning_rate=parameter_learning_rate,
               
                max_iter=parameter_max_iter,
                shuffle=parameter_shuffle ,
                random_state= parameter_random_state,
                
            )
            Neural.fit(X_train, Y_train)
            
            st.subheader('2. Model Performance')

            st.markdown('**2.1. Training set**')
            Y_pred_train = Neural.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_train, Y_pred_train) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )

            st.markdown('**2.2. Test set**')
            st.write("Prediction Score")
            Y_pred_test =Neural.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )

            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

            st.subheader('3. Model Parameters')
            st.write(Neural.get_params())
            
            st.subheader("4.Model Score")
            st.write("Return the mean accuracy on the given test data and labels.")
            st.write(Neural.score(X_test,Y_test))
            st.subheader("5.Model coefficients and intercept")
            st.info(len(Neural.coefs_))
            st.info(len(Neural.intercepts_[0]))
        except Exception as e:
            print(e)
            
            
      
