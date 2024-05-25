# Deep_Learning_Challenge

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

Table of Contents:
AlphabetSoupCharity_Optimization1.ipynb - This section includes the code implemented for the first model.
AlphabetSoupCharity_Optimization2.ipynb - This section includes the code implemented for the second model.
AlphabetSoupCharity_Optimization3.ipynb - This section includes the code implemented for the third model.
AlphabetSoupCharity_Optimization1.h5 - This section contains the output H5 file for the first model.
AlphabetSoupCharity_Optimization2.h5 - This section contains the output H5 file for the second model.
AlphabetSoupCharity_Optimization3.h5 - This section contains the output H5 file for the third model.
README.md - Contains Analysis and references for any sources used.

Data Preprocessing
What variable(s) are the target(s) for your model?

The target variable for the model is y, which contains the values we are trying to predict. Typically, this would be a column in your dataset that indicates the class or outcome you are trying to model, such as target in the example.
What variable(s) are the features for your model?

The features for the model are contained in X, which includes all the columns of the dataset except the target variable. These are the input variables that the model will use to make predictions.
What variable(s) should be removed from the input data because they are neither targets nor features?

Any identifier columns or metadata that do not provide information useful for the prediction should be removed. These could include IDs, names, or other non-informative attributes. For example, if the dataset contains a column ID or Name, these should be excluded from X.
Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?

Layers and Neurons:
First hidden layer: 64 neurons with tanh activation function and 50% dropout.
Second hidden layer: 32 neurons with relu activation function and 50% dropout.
Third hidden layer: 16 neurons with relu activation function.
Output layer: 1 neuron with sigmoid activation function.
Reasoning:
The tanh activation function in the first hidden layer helps in centering the data and making optimization easier.
The relu activation function in the subsequent layers is popular for its ability to avoid the vanishing gradient problem.
Dropout layers help in preventing overfitting by randomly dropping neurons during training.
The sigmoid activation function in the output layer is used for binary classification problems to output probabilities.
Were you able to achieve the target model performance?

The answer would depend on the actual results after training the model, which are not provided here. Generally, this involves evaluating the model's accuracy, precision, recall, or other relevant metrics on a validation or test set.
What steps did you take in your attempts to increase model performance?

Model Architecture: Experimented with different numbers of layers and neurons.
Regularization: Added dropout layers to prevent overfitting.
Activation Functions: Selected activation functions like tanh and relu to improve learning.
Hyperparameter Tuning: Adjusted the number of epochs, batch sizes, and learning rates.
Data Augmentation: If applicable, augmented the training data to provide more variability.
Feature Engineering: Scaled the features using StandardScaler for better performance of gradient descent.
Summary
Overall Results:

The deep learning model with three hidden layers (64, 32, and 16 neurons) and dropout regularization provides a robust framework for binary classification tasks.
The tanh and relu activation functions effectively capture complex patterns in the data.
The dropout layers help in reducing overfitting, thereby improving the generalizability of the model.
Recommendation for a Different Model:

Alternative Model: Consider using a Random Forest or Gradient Boosting model.
Explanation: These models are powerful ensemble methods that can handle non-linear relationships and interactions between features effectively. They also provide feature importance, which can offer insights into the most influential features in your dataset.
Benefits: Ensemble methods often perform well on structured data and can be easier to tune and interpret compared to deep neural networks. They also require less feature scaling and preprocessing.
By following these steps and making these adjustments, you can systematically improve your neural network model's performance and ensure that it is well-suited for your specific classification task.

References:
Tensorflow overview: https://www.tensorflow.org/tutorials?_gl=1*1a7vyot*_up*MQ..*_ga*MTUyNzI4MDA1NS4xNzE2NjU5Mjc0*_ga_W0YLR4190T*MTcxNjY1OTI3My4xLjAuMTcxNjY2MDYwMi4wLjAuMA..

Tensorflow .dropout overview: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

Assist in code and analysis: https://chatgpt.com/

Tips to improve accuracy: https://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/