# NumerML
Break the stock market maybe

## Todo first. 
- Pick a name and re-register for Numerai site. I'll give you that decision.

## Material

Site: https://numer.ai
Book: https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291

## Non-standard packages mostly required 
- scikit.learn (absolutely necessary)
- tensorflow 
- sonnet (DeepMind: https://github.com/deepmind/sonnet)
- TPOT (https://github.com/rhiever/tpot) 
- NumerAPI (https://github.com/numerai/NumerAPI)

## What
So the main idea here is really quite similar to what you were telling me about the intervention problem.  What's the best intervention?  Essentially, you can twiddle a couple of dials and come up with an answer by deriving some function that relates those dials to the observed outocmes.

So here there are currently 50 dials (it was only 33 a months ago). Each dial is a column in the .csv files.  The first three columns are metadata, the only important one is 'id' which is what you would think.  The next 50 are the dials or "features".  You can think of them as dials or, alternatively, as coordinates in 50 dimensional space.  At each of these points or dial settings there's an observation "target", either a 1 or a 0.  The purpose of this is to accurately determine the chance that any given point is a 1.

The model itself is not uploaded only the results, and it's scored on the probability accuracy (logloss), how different it is from other submissions (originality), and what percentage of predictions are true.

The binary target kind of limits the number of modeling schemes that can be used.  I think it'll require some kinds of pipelines and/or support vector machines.

### Nomenclature
Validation: A subset of the tournament data is for validation.  These dials are meant to more closely resemble the live data set. Validate the model by comparing it's predictions to the validation set target values.

Eras: As far as I can tell this can just be ignored.

Data_type: In the tournament file, there's a couple different data types.  Live and test data are what the model is evaluated on.

Parser contains a couple helper functions and a class parser that reads in the .csvs and splits them into the required data frames. Parser also has the getCSV function which uses the numeraiAPI to download and unzip the dataset. It also figures out the paths to the data and result files and stores them. Model takes a sci-kit regression model, trains it, applies it to the validation set, and prints the percentage of false results.

Any property that ends in an F is a feature dataframe and if it ends in a T it's a target data frame.  Valid, train, pred prefixed properties signify validation, training, and tournament data respectively, pred meaning predict the result of these features.

Dollar is an example script calling this stuff.  It's doing poorly now because it's only using simple Logistic Regression. 