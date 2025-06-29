import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd # Used for better printing of classification report details

# Load and filter dataset
data = load_iris()
X = data.data[data.target != 2]
y = data.target[data.target != 2]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define poisoned sample: slightly shift some feature values from class 0 towards class 1
num_poisoned = 5
class0_indices = np.where(y_train == 0)[0]
poison_indices = np.random.choice(class0_indices, num_poisoned, replace=False)

X_poison = X_train[poison_indices].copy()
y_poison = y_train[poison_indices].copy()

# Perturb features to mimic class 1
X_class1 = X_train[y_train == 1]
mean_class1 = X_class1.mean(axis=0)
X_poison += 0.4 * (mean_class1 - X_poison)

# Inject poisoned data
X_train_poisoned = np.vstack([X_train, X_poison])
y_train_poisoned = np.hstack([y_train, y_poison])  # same labels, no label flip

# Train models
model_clean = LogisticRegression(max_iter=1000, random_state=42) # Added random_state for reproducibility
model_poisoned = LogisticRegression(max_iter=1000, random_state=42)

model_clean.fit(X_train, y_train)
model_poisoned.fit(X_train_poisoned, y_train_poisoned)

# Evaluate
y_pred_clean = model_clean.predict(X_test)
y_pred_poisoned = model_poisoned.predict(X_test)

acc_clean = accuracy_score(y_test, y_pred_clean)
acc_poisoned = accuracy_score(y_test, y_pred_poisoned)

# Generate detailed classification reports
report_clean = classification_report(y_test, y_pred_clean, output_dict=True, target_names=['class 0', 'class 1'])
report_poisoned = classification_report(y_test, y_pred_poisoned, output_dict=True, target_names=['class 0', 'class 1'])

# Extract specific metrics for easy display in text
precision_clean_0 = report_clean['class 0']['precision']
recall_clean_0 = report_clean['class 0']['recall']
f1_clean_0 = report_clean['class 0']['f1-score']

precision_clean_1 = report_clean['class 1']['precision']
recall_clean_1 = report_clean['class 1']['recall']
f1_clean_1 = report_clean['class 1']['f1-score']

precision_poisoned_0 = report_poisoned['class 0']['precision']
recall_poisoned_0 = report_poisoned['class 0']['recall']
f1_poisoned_0 = report_poisoned['class 0']['f1-score']

precision_poisoned_1 = report_poisoned['class 1']['precision']
recall_poisoned_1 = report_poisoned['class 1']['recall']
f1_poisoned_1 = report_poisoned['class 1']['f1-score']

# Print the full classification reports for the console (optional for report, but good for debugging)
print("--- Classification Report for Clean Model ---")
print(classification_report(y_test, y_pred_clean, target_names=['class 0', 'class 1']))
print("\n--- Classification Report for Poisoned Model ---")
print(classification_report(y_test, y_pred_poisoned, target_names=['class 0', 'class 1']))

# --- Original Figure 1: Accuracy Comparison ---
fig1, ax1 = plt.subplots(figsize=(7, 5))
bars = ax1.bar(['Clean Data', 'Poisoned Data'], [acc_clean, acc_poisoned], color=['green', 'red'])
ax1.set_ylabel('Accuracy')
ax1.set_title('Effect of Clean-Label Data Poisoning on Model Accuracy')
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
plt.tight_layout()
plt.show()


# --- New Figure 2: Feature Distribution of Poisoned Samples vs. Original Classes ---
fig2, axes2 = plt.subplots(1, X.shape[1], figsize=(16, 4))
fig2.suptitle('Feature Distributions of Clean vs. Poisoned Samples (Training Data)')

for i in range(X.shape[1]): # Iterate through each feature
    # Distribution of clean class 0 and class 1
    axes2[i].hist(X_train[y_train == 0, i], bins=15, alpha=0.6, label='Clean Class 0', color='blue', density=True)
    axes2[i].hist(X_train[y_train == 1, i], bins=15, alpha=0.6, label='Clean Class 1', color='orange', density=True)
    # Mark the poisoned samples' feature values
    axes2[i].scatter(X_poison[:, i], np.zeros(num_poisoned), color='red', marker='x', s=100, label='Poisoned (Class 0 originally)', zorder=5) # y=0 for placing on x-axis
    axes2[i].set_title(f'Feature {i+1}')
    axes2[i].set_xlabel('Feature Value')
    if i == 0:
        axes2[i].set_ylabel('Density')
    axes2[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.show()


# --- New Figure 3: Decision Boundary Visualization ---
# For visualization, we'll use the first two features.
# You might want to choose the two features that show the most separation or impact.
feature_idx1, feature_idx2 = 0, 1 # Example: using sepal length and sepal width

# Create a meshgrid to plot the decision boundary
x_min, x_max = X[:, feature_idx1].min() - 0.5, X[:, feature_idx1].max() + 0.5
y_min, y_max = X[:, feature_idx2].min() - 0.5, X[:, feature_idx2].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict on the meshgrid for both models
# We need to create dummy values for the other two features as LogisticRegression expects 4 features
# For simplicity, we'll use the mean of the other features.
dummy_features_mean = np.mean(X[:, [2, 3]], axis=0) # Mean of features 2 and 3

def predict_on_mesh(model, xx, yy, dummy_features_mean):
    Z = np.c_[xx.ravel(), yy.ravel(),
              np.full(xx.size, dummy_features_mean[0]),
              np.full(xx.size, dummy_features_mean[1])]
    return model.predict(Z).reshape(xx.shape)

Z_clean = predict_on_mesh(model_clean, xx, yy, dummy_features_mean)
Z_poisoned = predict_on_mesh(model_poisoned, xx, yy, dummy_features_mean)


fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle(f'Decision Boundaries (using Features {feature_idx1+1} and {feature_idx2+1})')

# Plot for Clean Model
axes3[0].contourf(xx, yy, Z_clean, alpha=0.4, cmap=plt.cm.RdBu)
axes3[0].scatter(X_train[y_train == 0, feature_idx1], X_train[y_train == 0, feature_idx2],
                color='blue', label='Clean Train Class 0', edgecolors='k', s=50)
axes3[0].scatter(X_train[y_train == 1, feature_idx1], X_train[y_train == 1, feature_idx2],
                color='orange', label='Clean Train Class 1', edgecolors='k', s=50)
axes3[0].scatter(X_test[y_test == 0, feature_idx1], X_test[y_test == 0, feature_idx2],
                color='cyan', label='Clean Test Class 0', marker='^', s=50)
axes3[0].scatter(X_test[y_test == 1, feature_idx1], X_test[y_test == 1, feature_idx2],
                color='magenta', label='Clean Test Class 1', marker='^', s=50)
axes3[0].set_title('Clean Model Decision Boundary')
axes3[0].set_xlabel(f'Feature {feature_idx1+1}')
axes3[0].set_ylabel(f'Feature {feature_idx2+1}')
axes3[0].legend()


# Plot for Poisoned Model
axes3[1].contourf(xx, yy, Z_poisoned, alpha=0.4, cmap=plt.cm.RdBu)
axes3[1].scatter(X_train[y_train == 0, feature_idx1], X_train[y_train == 0, feature_idx2],
                color='blue', label='Clean Train Class 0', edgecolors='k', s=50)
axes3[1].scatter(X_train[y_train == 1, feature_idx1], X_train[y_train == 1, feature_idx2],
                color='orange', label='Clean Train Class 1', edgecolors='k', s=50)
axes3[1].scatter(X_test[y_test == 0, feature_idx1], X_test[y_test == 0, feature_idx2],
                color='cyan', label='Clean Test Class 0', marker='^', s=50)
axes3[1].scatter(X_test[y_test == 1, feature_idx1], X_test[y_test == 1, feature_idx2],
                color='magenta', label='Clean Test Class 1', marker='^', s=50)
# Highlight the poisoned samples on the plot
axes3[1].scatter(X_poison[:, feature_idx1], X_poison[:, feature_idx2],
                color='red', marker='X', s=200, label='Poisoned Samples', edgecolors='black', zorder=5)
axes3[1].set_title('Poisoned Model Decision Boundary')
axes3[1].set_xlabel(f'Feature {feature_idx1+1}')
axes3[1].set_ylabel(f'Feature {feature_idx2+1}')
axes3[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
plt.show()