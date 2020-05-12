# Sticker-Recommender
This is a project for DataX(2020 Spring)

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. modelSR.py - This contains code for our Machine Learning model to classify emtions(happy,sad,hate,love,surprise) bsed on training data in 'text_emotion_train_val_set.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited category based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter the text that they wanna predict

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python modelSR.py
```
This would create a serialized version of our model into a file modelSR.h5

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

Enter your input

If everything goes well, you should be able to see the recommended sticker on the HTML page!
