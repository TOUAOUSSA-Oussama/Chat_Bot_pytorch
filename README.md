**1/-Introduction**:

This ChatBot can answer easy questions as the model was trained on a small dataset that has only 7 categories with few examples.

**2/-Installation**:

The libraries and dependencies are available in “requirement.txt”. The file “nltk_utils” has three functions to prepare the data. “train.py” does the preprocessing of the dataset, defines the hyperparameters, creates the model defined in “model.py” and trains it. The model is implemented in Pytorch and it’s a feedforward model that contains 3 layers. The “data.pth” file contains the pre-trained model, the data after the preprocessing, and some necessary hyperparameters.

**3/-Usage**:

To run the ChatBot, you just need to execute the file “chat.py” and then communicate with the bot.
