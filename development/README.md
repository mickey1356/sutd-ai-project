# Development

This folder holds the necessary scripts and notebooks used to obtain data, as well as train and test the sentiment-price model.

## Information

Broadly speaking, we followed the standard process of (1) obtaining data, (2) labelling data, (3) cleaning data, (4) training model, and finally (5) deploying.

The deployment scripts are further detailed in the `deployment` folder.

We further detail each development step below.

The notebooks were run using Google Colab notebooks.

### 1. Obtaining Data

`scrape-reddit.py`

We used both [Pushshift](https://github.com/pushshift/api) as well as a Python wrapper for Reddit's official API, [PRAW](https://github.com/praw-dev/praw), in order to obtain the post data from r/wallstreetbets.

Pushshift was used because the Reddit's API does not support searching for posts by a specified date. However, Pushshift was also limited in that they do not update important information such as the score and number of comments, and hence PRAW was used to get the updated information from Reddit.

`get-stock-data.ipynb`

We used [yfinance](https://github.com/ranaroussi/yfinance) to obtain daily price data for the tickers we were interested in (BANG stocks). [Yahoo! finance](https://finance.yahoo.com/) offers free historical price data, and the yfinance library allows one to easily obtain such market data from Yahoo.

`get-sent-data.ipynb`

As we were exploring how the sentiments of posts on r/wallstreetbets would affect the price movements of certain tickers, we needed a reliable method to obtain sentiment data. We test out [VADER](https://github.com/cjhutto/vaderSentiment), a lexicon and rule-based sentiment analysis tool supposedly attuned to sentiments expressed in social media. We took a sample of its predictions on the post data obtained from Reddit for manual verification.

`sent-training.ipynb`

Because VADER proved to be inaccurate on r/wallstreetbets post data, we decided to build a basic sentiment model.

`sent-deploy.ipynb`

We demonstrate the effectiveness of our model in predicting sentiments by passing in a few sample text data.

###2. Labelling and Cleaning Data

`data-integration.ipynb`

We generate the target data for our model, as well as perform additional filtering and cleaning of the data, such that it falls in line with how we would expect the model to be deployed and run.

###3. Model Training

`train-nn.ipynb`

We make use of PyTorch to build a simple Neural Network model, and train it on the data we have previously generated.

###4. Deployment

The scripts and notebooks would produce two files: `sentiment.pkl` and `bang_model`. These two files should be moved to deployment folder, and should be located in the same directory as the deployment script, `deploy.py`.

##5. Additional Requirements

You would need to create a Firebase database and fill up the `.env` file with the necessary credentials in order to fully run `data-integration.ipynb`.

