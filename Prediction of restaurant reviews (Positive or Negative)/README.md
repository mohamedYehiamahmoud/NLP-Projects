# Restaurant Review Sentiment Analysis

This project focuses on sentiment analysis of restaurant reviews using Natural Language Processing (NLP) techniques. It aims to classify restaurant reviews as positive or negative based on the text content.

## Project Structure

- `my_natural_language_processing.py`: Main Python script for NLP processing and model training
- `Restaurant_Reviews.tsv`: Dataset containing restaurant reviews and their sentiment labels

## Features

- Text preprocessing using NLTK
- Bag of Words model for feature extraction
- Naive Bayes classifier for sentiment prediction
- Train-test split for model evaluation

## Dependencies

- numpy
- matplotlib
- pandas
- nltk
- scikit-learn

## Usage

1. Ensure all dependencies are installed
2. Place the `Restaurant_Reviews.tsv` file in the same directory as the script
3. Run `my_natural_language_processing.py`

## Data

The `Restaurant_Reviews.tsv` file contains 1000 restaurant reviews. Each review has two columns:
- Review: The text of the restaurant review
- Liked: Binary sentiment label (1 for positive, 0 for negative)

## Model Performance

The model's performance is evaluated using a confusion matrix. Check the script output for detailed results.

## Future Improvements

- Experiment with different classifiers (e.g., SVM, Random Forest)
- Implement cross-validation for more robust evaluation
- Try advanced NLP techniques like word embeddings or transformer models

## Contributing

Feel free to fork this repository and submit pull requests to contribute to this project. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)