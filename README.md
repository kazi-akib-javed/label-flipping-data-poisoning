# label-flipping-data-poisoning
This code demonstrates a clean-label data poisoning attack against a Logistic Regression model trained on a binary classification subset of the Iris dataset. The core idea of clean-label poisoning is to manipulate the features of a small number of training samples, subtly shifting them towards another class's characteristics, while preserving their original, correct labels. This contrasts with traditional poisoning where labels are flipped, making it harder to detect.

The experiment proceeds by:

    Loading and preparing the Iris dataset for a binary classification task (distinguishing between two classes).

    Splitting the data into training and testing sets.

    Crafting a small set of "poisoned" samples. These samples, originally belonging to one class, have their feature values slightly perturbed to resemble the mean features of the other class, but their true class labels remain unchanged.

    Training two Logistic Regression models: one on the original, clean training data, and another on the training data augmented with these clean-label poisoned samples.

    Evaluating both models on the unseen test set using various metrics including overall accuracy, precision, recall, and F1-score for each class.

    Visualizing the results through three key figures:

        A bar chart comparing the overall test accuracy of the clean and poisoned models.

        Histograms illustrating the feature distributions of clean training data for both classes, with markers highlighting the location of the injected poisoned samples. This shows how poisoned samples, despite their correct labels, occupy an ambiguous or misleading region in the feature space.

        Decision boundary plots for both the clean and poisoned models (projected onto two key features). These plots visually demonstrate how the poisoned data can subtly shift or distort the model's decision boundary, leading to misclassifications on legitimate test samples, thereby explaining the observed performance degradation.

The objective is to showcase how an attacker can degrade model performance by injecting seemingly innocuous (correctly labeled) but strategically altered data points into the training set, highlighting the vulnerability of machine learning models to this type of adversarial attack.