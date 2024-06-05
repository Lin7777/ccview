import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pydotplus
from tqdm import tqdm
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
import matplotlib.font_manager as fm

# Load data
file_path = 'cc_data.xlsx'
data = pd.read_excel(file_path)

# Label risky crew members (total ranking percentage >= 0.80)
data['Risky_Crew'] = (data['总排名百分比'] >= 0.80).astype(int)

# Select features and exclude irrelevant columns
features = data.drop(columns=['工号', '姓名', 'Risky_Crew', '总排名', '总排名百分比'])
target = data['Risky_Crew']

# Encode categorical variables
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

# Save LabelEncoders
joblib.dump(label_encoders, 'label_encoders.pkl')

# Standardize numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save StandardScaler
joblib.dump(scaler, 'scaler.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'MLP': MLPClassifier(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True)
}

# Train and evaluate models
results = {}
for model_name, model in tqdm(models.items(), desc="Training models"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'classification_report': report
    }
    # Save models
    joblib.dump(model, f'{model_name.lower().replace(" ", "_")}_model.pkl')

# Output results
for model_name, result in results.items():
    print(f"{model_name} Model:")
    print(f"Accuracy: {result['accuracy']}")
    print(f"Classification Report:\n{result['classification_report']}")

# Analyze feature importance
feature_importances = {}
for model_name, result in results.items():
    model = result['model']
    if hasattr(model, 'coef_'):  # Logistic Regression, SVM
        importances = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):  # Decision Tree, Random Forest
        importances = model.feature_importances_
    else:
        continue  # Skip models without feature importances
    feature_importances[model_name] = importances

# Output feature importance
features_list = features.columns
for model_name, importances in feature_importances.items():
    print(f"\n{model_name} Feature Importances:")
    for feature, importance in zip(features_list, importances):
        print(f"{feature}: {importance}")

# Grid Search for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)

# Get the best Decision Tree model
best_dt_model = grid_search_dt.best_estimator_

# Evaluate the best Decision Tree model
y_pred_best_dt = best_dt_model.predict(X_test)
accuracy_best_dt = accuracy_score(y_test, y_pred_best_dt)
report_best_dt = classification_report(y_test, y_pred_best_dt)

print(f"Best Decision Tree Model Accuracy: {accuracy_best_dt}")
print(f"Best Decision Tree Classification Report:\n{report_best_dt}")

# Save the best Decision Tree model
joblib.dump(best_dt_model, 'best_decision_tree_model.pkl')

# Visualize the Decision Tree
dot_data = export_graphviz(best_dt_model, out_file=None,
                           feature_names=features_list,
                           class_names=['Not Risky', 'Risky'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_jpg("decision_tree.jpg")  # This will save the tree as a file named "decision_tree.jpg"
Image(filename="decision_tree.jpg")

# Calculate correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap with Chinese font support
plt.figure(figsize=(12, 10))

# Set the font to a Chinese font
chinese_font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # Update the path to your Chinese font

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"fontproperties": chinese_font})
plt.title('Correlation Matrix', fontproperties=chinese_font)
plt.xticks(fontproperties=chinese_font)
plt.yticks(fontproperties=chinese_font)
plt.show()
