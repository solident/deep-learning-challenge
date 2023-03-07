## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special considerations for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

### Report 

1. **Overview** of the analysis: Explain the purpose of this analysis.

The objective of this study is to utilize a deep learning neural network to develop a binary classifier for predicting the likelihood of success for funding applicants from Alphabet Soup, a nonprofit organization. The dataset provided contains information on more than 34,000 organizations, including metadata such as application type, industry sector affiliation, government organization classification, funding use case, income classification, funding amount requested, and whether the funds were used efficiently. The analysis involves preprocessing the data by removing redundant columns, encoding categorical variables, and dividing the dataset into training and testing sets. The neural network model is then created, trained, and assessed to determine its loss and accuracy. Subsequently, the model is optimized using a range of methods, such as adjusting input data, incorporating additional neurons and hidden layers, employing different activation functions, and tweaking the number of epochs. The overarching aim is to achieve a predictive accuracy of over 75% and preserve the optimized model as an HDF5 file.

2. **Results**: Using bulleted lists and images to support your answers, address the following questions:

    * The target variable(s) that the model will predict is IS_SUCCESSFUL, as it is the binary classification outcome variable that indicates whether a charity donation was successful or not.

* Data Preprocessing

    * What variable(s) are the target(s) for your model?
        * The target variable(s) that the model will predict is IS_SUCCESSFUL, as it is the binary classification outcome variable that indicates whether a charity donation was successful or not.
    * What variable(s) are the features for your model?
        * The feature variables are all the other columns in the DataFrame, except for 'IS_SUCCESSFUL'.
    * What variable(s) should be removed from the input data because they are neither targets nor features?
        * During the preprocessing stage of the analysis, it was discovered that the Employee Identification Number (EIN) does not provide relevant information for our predictive model. As a result, the EIN variable was excluded from the feature and target selection process, as it is not relevant to the analysis.
    
* Compiling, Training, and Evaluating the Model

    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    
    I opted to employ three hidden layers with 20, 27, and 3 neurons respectively for this neural network model. This specific combination was chosen after performing numerous iterations and tests with various numbers of neurons and layers. After thorough experimentation, it was discovered that this particular configuration produced the most favorable outcomes in terms of accuracy and loss.

    In regards to activation functions, I utilized ReLU for the first hidden layer to introduce non-linearity into the model and enhance its performance. For the second and third hidden layers, I selected sigmoid as it is better suited for binary classification tasks like the one at hand. Finally, for the output layer, I implemented the sigmoid activation function to guarantee that the output falls between 0 and 1, which is necessary for binary classification.

    * Were you able to achieve the target model performance?
    
    I managed to attain a predictive accuracy that exceeded 75%.

    * What steps did you take in your attempts to increase model performance?

    In the course of optimizing the model, the EIN column was initially removed because it was not deemed as either a target or feature. However, after numerous attempts to optimize the model, it was discovered that retaining the NAME column actually improved the model's accuracy. To that end, a cutoff value was set, and a list of names was created to replace those that appeared infrequently using value counts. Any name that appeared less than 10 times was substituted with "Other". In a similar vein, a cutoff value was selected for the CLASSIFICATION column, and any classification that occurred less than 2000 times was replaced with "Other". The binning process was checked to verify its effectiveness.
    
**Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

    To recap, the implementation of TensorFlow and Keras to construct a deep learning model was able to classify the success of Alphabet Soup-funded organizations based on their features with a predictive accuracy of 78%. The model underwent various optimization attempts, such as column dropping, categorical variable binning, hidden layer and neuron addition, and the exploration of different activation functions, among other tweaks. Although the target predictive accuracy of 75% was ultimately attained, it necessitated numerous optimization attempts to achieve that threshold.
  
