"""
visualize.py

Visualization of prediction confidence and sentiment distribution.
"""

import matplotlib.pyplot as plt
from sentiment_analysis import SentimentAnalyzer


def visualize_predictions():
    analyzer = SentimentAnalyzer()

    texts = [
        "I love this product!",
        "This is horrible.",
        "It's fine, nothing special.",
        "Absolutely amazing experience!",
        "Worst service ever."
    ]

    predictions = analyzer.batch_predict(texts)

    labels = [p["label"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]

    positive_count = labels.count("POSITIVE")
    negative_count = labels.count("NEGATIVE")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(["Positive", "Negative"], [positive_count, negative_count])
    plt.title("Sentiment Distribution")

    plt.subplot(1, 2, 2)
    plt.hist(confidences, bins=5)
    plt.title("Confidence Score Distribution")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_predictions()