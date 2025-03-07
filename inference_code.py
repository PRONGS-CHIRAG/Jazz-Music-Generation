def inference_model(LSTM_cell, densor, n_x = 90, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_x -- number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_x))
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    #Create an empty list of "outputs" to  store your predicted values
    outputs = []
    #Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        #Perform one step of LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        #Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)
        #Append the prediction "out" to "outputs"
        outputs.append(out)
        #Set the prediction "out" to be the next input "x". You will need to use RepeatVector(1).
        x = RepeatVector(1)(out)
    #Create model instance with the correct "inputs" and "outputs"
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    return inference_model


inference_model = inference_model(LSTM_cell, densor)  # Create an inference model using LSTM_cell and densor components
x1 = np.zeros((1, 1, 90))  # Initialize input tensor with zeros (batch_size=1, sequence_length=1, features=90)
x1[:,:,35] = 1  # Set the 36th feature (index 35) to 1, creating a one-hot encoded input
a1 = np.zeros((1, n_a))  # Initialize hidden state tensor with zeros (batch_size=1, hidden_units=n_a)
c1 = np.zeros((1, n_a))  # Initialize cell state tensor with zeros (batch_size=1, hidden_units=n_a)
predicting = inference_model.predict([x1, a1, c1])  # Run inference with the prepared input tensors
indices = np.argmax(predicting, axis = -1)  # Find the index with highest probability for each prediction
results = to_categorical(indices, num_classes=90)  # Convert indices to one-hot encoded vectors with 90 classes