class Anamoly:
    import torch

    MODEL_SELECTED = globals()
    #MODEL_SELECTED = "deepcnn" # Possible Values ['deepcnn', 'lstmaenn']
    LOOKBACK_SIZE = globals()
    #LOOKBACK_SIZE = 10

    def set_config(**kwargs):
        """
        Select Model & Window Size : Function to select models from Deep CNN & LSTMAE with Possible Values ['deepcnn', 'lstmaenn'], and time window size
        """
        for key, value in kwargs.items():
            print("{0} = {1}".format(key, value))

        MODEL_SELECTED = list(kwargs.values())[0]
        LOOKBACK_SIZE = list(kwargs.values())[1]

        return MODEL_SELECTED, LOOKBACK_SIZE # kwargs
        #MODEL_SELECTED = MODEL_INPUT
        #LOOKBACK_SIZE = WINDOW_SIZE
        #return MODEL_SELECTED, LOOKBACK_SIZE

    assert MODEL_SELECTED == MODEL_SELECTED
    assert LOOKBACK_SIZE == LOOKBACK_SIZE

    def read_data(data, column_index_to_drop:int,timestamp_column_index:int): #data_file
        """
        Data ingestion : Function to read and formulate the data
        """
        import pandas as pd
        data.drop(data.columns[column_index_to_drop], inplace=True, axis=1) # 
        df = data.copy()
        data.set_index(data.columns[timestamp_column_index], inplace=True) # LOCAL_DATE
        data.index = pd.to_datetime(data.index)
        data.fillna(0, inplace=True)
        return data, df


    def data_pre_processing(df, LOOKBACK_SIZE=LOOKBACK_SIZE):
        """
        Data pre-processing : Function to create data for Model
        """
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        try:
            scaled_data = MinMaxScaler(feature_range = (0, 1))
            data_scaled_ = scaled_data.fit_transform(df)
            df.loc[:,:] = data_scaled_
            _data_ = df.to_numpy(copy=True)
            X = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,LOOKBACK_SIZE,df.shape[1]))
            X_data = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,df.shape[1]))
            Y = np.zeros(shape=(df.shape[0]-LOOKBACK_SIZE,df.shape[1]))
            timesteps = []
            for i in range(LOOKBACK_SIZE-1, df.shape[0]-1):
                timesteps.append(df.index[i])
                Y[i-LOOKBACK_SIZE+1] = _data_[i+1]
                for j in range(i-LOOKBACK_SIZE+1, i+1):
                    X[i-LOOKBACK_SIZE+1][LOOKBACK_SIZE-1-i+j] = _data_[j]
                    X_data[i-LOOKBACK_SIZE+1] = _data_[j]
            return X,Y,timesteps, X_data
        except Exception as e:
            print("Error while performing data pre-processing : {0}".format(e))
            return None, None, None, None


    class DeepCNN(torch.nn.Module):
        """
        Model : Class for DeepCNN model
        """
        def __init__(self, LOOKBACK_SIZE, DIMENSION):
            super(Anamoly.DeepCNN, self).__init__()
            import torch
            self.conv1d_1_layer = torch.nn.Conv1d(in_channels=LOOKBACK_SIZE, out_channels=16, kernel_size=2) # , stride=2
            self.relu_1_layer = torch.nn.ReLU()
            self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=1)
            self.conv1d_2_layer = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2)
            self.relu_2_layer = torch.nn.ReLU()
            self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=1)
            self.flatten_layer = torch.nn.Flatten()
            self.dense_1_layer = torch.nn.Linear(80, 40)
            self.relu_3_layer = torch.nn.ReLU()
            self.dropout_layer = torch.nn.Dropout(p=0.25)
            self.dense_2_layer = torch.nn.Linear(40, DIMENSION)
            
        def forward(self, x):
            x = self.conv1d_1_layer(x)
            x = self.relu_1_layer(x)
            x = self.maxpooling_1_layer(x)
            x = self.conv1d_2_layer(x)
            x = self.relu_2_layer(x)
            x = self.maxpooling_2_layer(x)
            x = self.flatten_layer(x)
            x = self.dense_1_layer(x)
            x = self.relu_3_layer(x)
            x = self.dropout_layer(x)
            return self.dense_2_layer(x)


    class LSTMAENN(torch.nn.Module):
        """
            Model : Class for LSTMAENN model
        """
        def __init__(self, LOOKBACK_SIZE, DIMENSION):
            import torch
            super(Anamoly.LSTMAENN, self).__init__()
            self.lstm_1_layer = torch.nn.LSTM(DIMENSION, 128, 1)
            self.dropout_1_layer = torch.nn.Dropout(p=0.2)
            self.lstm_2_layer = torch.nn.LSTM(128, 64, 1)
            self.dropout_2_layer = torch.nn.Dropout(p=0.2)
            self.lstm_3_layer = torch.nn.LSTM(64, 64, 1)
            self.dropout_3_layer = torch.nn.Dropout(p=0.2)
            self.lstm_4_layer = torch.nn.LSTM(64, 128, 1)
            self.dropout_4_layer = torch.nn.Dropout(p=0.2)
            self.linear_layer = torch.nn.Linear(128, DIMENSION)
            
        def forward(self, x):
            x, (_,_) = self.lstm_1_layer(x)
            x = self.dropout_1_layer(x)
            x, (_,_) = self.lstm_2_layer(x)
            x = self.dropout_2_layer(x)
            x, (_,_) = self.lstm_3_layer(x)
            x = self.dropout_3_layer(x)
            x, (_,_) = self.lstm_4_layer(x)
            x = self.dropout_4_layer(x)
            return self.linear_layer(x)


    def make_train_step(model, loss_fn, optimizer):
        """
        Computation : Function to make batch size data iterator
        """
        def train_step(x, y):
            import torch
            torch.multiprocessing.freeze_support()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.train()
            yhat = model(x).to(device)
            loss = loss_fn(y, yhat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item()
        return train_step


    def compute(X,Y, LOOKBACK_SIZE, num_of_numerical_features:int, MODEL_SELECTED=MODEL_SELECTED):
        """
            Computation : Find Anomaly using model based computation 
        """
        import torch
        import numpy as np
        torch.multiprocessing.freeze_support()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        if str(MODEL_SELECTED) == "lstmaenn":
            model = Anamoly.LSTMAENN(LOOKBACK_SIZE,num_of_numerical_features).to(device) # 26   LSTMAENN(10,7).to(device)
            print(model)
            criterion = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(X.astype(np.float32)))
            train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
            train_step = Anamoly.make_train_step(model, criterion, optimizer)
            for epoch in range(30):
                loss_sum = 0.0
                ctr = 0
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    loss_train = train_step(x_batch, y_batch)
                    loss_sum += loss_train
                    ctr += 1
                print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum/ctr), epoch+1))
            hypothesis = model(torch.from_numpy(X.astype(np.float32)).to(device)).detach().cpu().numpy()
            loss = np.linalg.norm(hypothesis - X, axis=(1,2))
            return loss.reshape(len(loss),1), train_data, model
        elif str(MODEL_SELECTED) == "deepcnn":
            model = Anamoly.DeepCNN(LOOKBACK_SIZE,num_of_numerical_features).to(device) # 26    DeepCNN(10, 7).to(device)
            print(model)
            criterion = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)
            train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.float32)))
            train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
            train_step = Anamoly.make_train_step(model, criterion, optimizer)
            for epoch in range(30):
                loss_sum = 0.0
                ctr = 0
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    loss_train = train_step(x_batch, y_batch)
                    loss_sum += loss_train
                    ctr += 1
                print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum/ctr), epoch+1))
            hypothesis = model(torch.from_numpy(X.astype(np.float32)).to(device)).detach().cpu().numpy()
            loss = np.linalg.norm(hypothesis - Y, axis=1)
            return loss.reshape(len(loss),1), train_data, model
        else:
            print("Selection of Model is not in the set")
            return None

    def find_anamoly(loss, T):
        """
        Compute Anamoly Confidence Score
        """
        import pandas as pd
    
        loss_df = pd.DataFrame(loss, columns = ["loss"])
        loss_df.index = T
        loss_df.index = pd.to_datetime(loss_df.index)
        loss_df["timestamp"] = T
        loss_df["timestamp"] = pd.to_datetime(loss_df["timestamp"])
        loss_df.to_csv('anamoly_loss_df.csv')
        return loss_df


    def plot_anamoly_results(loss_df):
        """
        Anamoly Visualization 
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,10))
        sns.set_style("darkgrid")
        ax = sns.distplot(loss_df["loss"], bins=100, label="Frequency")
        ax.set_title("Frequency Distribution | Kernel Density Estimation")
        ax.set(xlabel='Anomaly Confidence Score', ylabel='Frequency (sample)')
        plt.axvline(1.80, color="k", linestyle="--")
        plt.legend()


        plt.figure(figsize=(20,10))
        ax = sns.lineplot(x="timestamp", y="loss", data=loss_df, color='g', label="Anomaly Score")
        ax.set_title("Anomaly Confidence Score vs Timestamp")
        ax.set(ylabel="Anomaly Confidence Score", xlabel="Timestamp")
        plt.legend()


    def explainable_results(specific_prediction_sample_to_explain:int, X, Y, input_label_index_value):  # , anamoly_data
        """
        Understand, interpret, and trust the results on the deep models at individual/samples level
        """
        '''
        from sklearn.ensemble import RandomForestRegressor
        import xgboost
        import shap
        import numpy as np
        shap.initjs()

        y = [max(i) for i in Y]

        my_model_1 = xgboost.XGBRegressor().fit(X, np.array(y))

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer_xgb = shap.Explainer(my_model_1)
        shap_values_xgb = explainer_xgb(anamoly_data)

        # visualize the first prediction's explanation
        shap.plots.waterfall(shap_values_xgb[specific_prediction_sample_to_explain])

        
        my_model_2 = RandomForestRegressor(random_state=0).fit(X, np.array(y))

        data_for_prediction = X[specific_prediction_sample_to_explain]  # use 1 row of data here. Could use multiple rows if desired
        # Create object that can calculate shap values
        explainer_rf = shap.TreeExplainer(my_model_2)

        # Calculate Shap values
        shap_values = explainer_rf.shap_values(data_for_prediction)

        shap.force_plot(explainer_rf.expected_value[specific_prediction_sample_to_explain], shap_values[1], data_for_prediction)
        '''

        # Quick Clean Hack Suggested by - Cory Randolph @coryroyce
        import shap
        import numpy as np
        import pandas as pd
        from keras.models import Sequential
        from keras.layers import Dense
        import ipywidgets as widgets

        # Get the number of inputs and outputs from the dataset
        n_inputs, n_outputs = X.shape[1], Y.shape[1]

        def get_model(n_inputs, n_outputs):
            model_nn = Sequential()
            model_nn.add(Dense(32, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
            model_nn.add(Dense(n_outputs, kernel_initializer='he_uniform'))
            model_nn.compile(loss='mae', optimizer='adam')
            return model_nn

        model_nn = get_model(n_inputs, n_outputs)

        model_nn.fit(X.iloc[10:,:].values, Y, epochs=30)

        model_nn.evaluate(x = X.iloc[10:,:].values, y = Y)

        XpredictInputData = X.iloc[specific_prediction_sample_to_explain,:] # X[specific_prediction_sample_to_explain,:]

        if (XpredictInputData.ndim == 1):
            XpredictInputData = np.array([XpredictInputData])

        print(model_nn.predict(XpredictInputData)) # 0:1

        '''
        Here we take the Keras model trained above and explain why it makes different predictions on individual samples.

        Set the explainer using the Kernel Explainer (Model agnostic explainer method form SHAP).
        '''
        explainer = shap.KernelExplainer(model = model_nn.predict, data = X.head(50), link = "identity") # data = X[0:50]

        '''
        Get the Shapley value for a single example.
        '''
        # Set the index of the specific example to explain

        shap_value_single = explainer.shap_values(X = X.iloc[specific_prediction_sample_to_explain,:], nsamples = 100)  # X[specific_prediction_sample_to_explain,:]

        '''
        Display the details of the single example
        '''
        print(X.iloc[specific_prediction_sample_to_explain,:]) 
        '''
        Choose the label/output/target to run individual explanations on:

        Note: The dropdown menu can easily be replaced by manually setting the index on the label to explain.
        '''
        # Create the list of all labels for the drop down list
        label_cols = ['window_diff_0', 'window_diff_1', 'window_diff_2', 'window_diff_3', 'window_diff_4', 'window_diff_5', 'window_diff_6']
        df_labels = pd.DataFrame(data = Y, columns = label_cols)
        df_labels.to_csv('y_labels.csv')
        list_of_labels = df_labels.columns.to_list() # Y.columns.to_list()

        # Create a list of tuples so that the index of the label is what is returned
        tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

        # Create a widget for the labels and then display the widget
        current_label = widgets.Dropdown(options=tuple_of_labels,
                                      value=input_label_index_value,
                                      description='Select Label:'
                                      )

        # Display the dropdown list (Note: access index value with 'current_label.value')
        print(current_label)
        #Dropdown(description='Select Label:', options=(('labels_01', 0), ('labels_02', 1), ('labels_03', 2), etc

        '''
        Plot the force plot for a single example and a single label/output/target
        '''
        print(f'Current label Shown: {list_of_labels[current_label.value]}')

        # print the JS visualization code to the notebook
        shap.initjs()

        shap.force_plot(base_value = explainer.expected_value[current_label.value],
                        shap_values = shap_value_single[current_label.value], 
                        features = X.iloc[specific_prediction_sample_to_explain,:] # X_idx:X_idx+1
                        )

        '''
        Create the summary plot for a specific output/label/target.
        '''
        # Note: We are limiting to the first 50 training examples since it takes time to calculate the full number of sampels
        shap_values = explainer.shap_values(X = X.iloc[0:50,:], nsamples = 100) # X[0:50,:]

        print(f'Current Label Shown: {list_of_labels[current_label.value]}\n')

        # print the JS visualization code to the notebook
        shap.initjs()

        shap.summary_plot(shap_values = shap_values[current_label.value],
                  features = X.iloc[0:50,:] # X[0:50,:]
                  )

        '''
        Force Plot for the first 50 individual examples.
        '''
        print(f'Current Label Shown: {list_of_labels[current_label.value]}\n')

        # print the JS visualization code to the notebook
        shap.initjs()

        shap.force_plot(base_value = explainer.expected_value[current_label.value],
                        shap_values = shap_values[current_label.value], 
                        features = X.iloc[0:50,:] # X[0:50,:]
                        )




