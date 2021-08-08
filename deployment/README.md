# Deployment

This folder holds the script used to deploy the model.

## Information

`deploy.py` can be deployed on the cloud using (for example) a Google Cloud Platform instance. It would schedule three different tasks to be run at different timings.

The first task would run at the 20 minute mark every hour, and it will scrape Reddit to obtain and store post data.

The second task would run at 1.30pm GMT on weekdays, and it will generate the input data (based on the stored post data) and run it through the provided model. It will then upload the results on Firebase.

The final task would run at 10pm GMT on weekdays (after the market closes), and it will retrain the model on the day's data in order to ensure robustness, as well as allow the model to improve over time.

## Additional Requirements

In order to properly deploy the model, you would require certain credentials from Reddit and Firebase as specified in the `.env` file.

Firstly, you would need to create a Reddit account, and create a Reddit app (https://www.reddit.com/prefs/apps) in order to obtain the appropriate Reddit-related credentials.

Then, you would need to create a Firebase real-time database storage, in order to obtain the Firebase-related information.

Finally, you would need to replace the `<to insert>` fields in the `.env` file with your user-specific credentials.

