# Sticker-Recommender
This is a project for 290 DataX(Spring 2020)

### Prerequisites
1. Stickers we used were scraped from line offical stickers: https://store.line.me/stickershop/home/general/en
2. You must have Scikit Learn, Pandas (for Machine Leraning Model),Flask (for API) and virtualenv(for environment)installed.

### Project Structure
This project has four major parts :
1. modelSR.py - This contains code for our Machine Learning model to classify emtions(happy,sad,hate,love,surprise) bsed on training data in 'text_emotion_train_val_set.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the predicted emtion based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter the text that they wanna predict

### Running the project
1. Ensure that you are in the project home directory. Create a virtual environment.by running below command -
```
virtualenv env
```
2. activate the virtual environment by running below command -
```
source env/bin/activate
```
for windows
```
\env\Scripts\activate.bat
```
3. create the machine learning model by running below command -
```
python modelSR.py
```
This would create a serialized version of our model into a file modelSR.h5

4. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

5. Navigate to URL http://localhost:5000

Enter your input

If everything goes well, you should be able to see the recommended sticker on the HTML page!

### Reference
1. Learn Flask for Python - Full Tutorial: https://www.youtube.com/watch?v=Z1RJmh_OqeA&t=269s
2. Deploy Machine Learning Model using Flask: https://www.youtube.com/watch?v=UbCWoMf80PY&t=579s
3. UC Berkeley 290 Data-X: https://data-x.blog/
