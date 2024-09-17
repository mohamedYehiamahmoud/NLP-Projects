# IMDB Movie Reviews Sentiment Analysis

This project performs sentiment analysis on IMDB movie reviews using machine learning techniques. It classifies movie reviews as positive or negative based on their content.

## Project Structure

- `Movie Reviews Sentiment Analysis.py`: Main Python script for data processing, model training, and evaluation
- `IMDB Dataset.csv`: Dataset containing IMDB movie reviews and their sentiment labels (not included in the repository)

## Features

- Data cleaning and preprocessing
- TF-IDF vectorization for feature extraction
- Support Vector Machine (SVM) classifier for sentiment prediction
- Train-test split for model evaluation

## Dependencies

- pandas
- scikit-learn

## Usage

1. Ensure all dependencies are installed
2. Place the `IMDB Dataset.csv` file in the same directory as the script
3. Run `Movie Reviews Sentiment Analysis.py`

## Data

The `IMDB Dataset.csv` file should contain movie reviews and their corresponding sentiment labels. The script expects two columns:
- review: The text of the movie review
- sentiment: The sentiment label ('positive' or 'negative')

## Model Performance

The model's performance is evaluated using accuracy score. The script will print the accuracy of the model on the test set.

## Data Preprocessing

The script performs the following preprocessing steps:
1. Removes duplicate reviews
2. Converts text to lowercase
3. Removes special characters and URLs from reviews
4. Applies TF-IDF vectorization

## Future Improvements

- Experiment with different classifiers (e.g., Naive Bayes, Random Forest)
- Implement cross-validation for more robust evaluation
- Try advanced NLP techniques like word embeddings or transformer models
- Perform hyperparameter tuning to optimize model performance

## Contributing

Feel free to fork this repository and submit pull requests to contribute to this project. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)