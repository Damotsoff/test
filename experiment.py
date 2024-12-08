#experimetal code for testing the performance of the model


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from river.ensemble import AdaptiveRandomForestClassifier
from river.metrics import Accuracy
import pandas as pd

def generate_concept_drift_data(n_samples=5000, n_features=10, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=5,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    data['target'] = y
    return data

def create_stream_with_drift(data, drift_point=1000):
    stream_before_drift = data.iloc[:drift_point].copy()
    stream_after_drift = data.iloc[drift_point:].copy()
    
    # Simulate concept drift
    stream_after_drift.loc[:, 'feature_0'] = stream_after_drift['feature_0'] * 1.5
    stream_after_drift.loc[:, 'target'] = 1 - stream_after_drift['target']
    
    return stream_before_drift, stream_after_drift

def run_experiment():
    # Generate data
    data = generate_concept_drift_data()
    
    # Create streams
    stream_before_drift, stream_after_drift = create_stream_with_drift(data)
    
    # Initialize models
    models = {
        'SGDClassifier': SGDClassifier(random_state=42),
        'GaussianNB': GaussianNB(),
        'AdaptiveRandomForest': AdaptiveRandomForestClassifier(
            n_models=10,
            seed=42
        )
    }
    
    # Initialize accuracy lists
    accuracy_sgd = []
    accuracy_nb = []
    accuracy_rf = []
    
    # Train models on initial data
    X_train = stream_before_drift.drop('target', axis=1)
    y_train = stream_before_drift['target']
    
    # Initial training for sklearn models
    for model_name, model in models.items():
        if model_name != 'AdaptiveRandomForest':
            if model_name == 'SGDClassifier':
                model.partial_fit(X_train, y_train, classes=np.unique(data['target']))
            else:
                model.fit(X_train, y_train)
        else:
            # Train River model
            for idx, row in stream_before_drift.iterrows():
                x = row.drop('target').to_dict()
                y = row['target']
                model.learn_one(x, y)
    
    # Evaluate models over time with sliding windows
    window_size = 50
    
    for i in range(0, len(stream_after_drift), window_size):
        window = stream_after_drift.iloc[i:i+window_size]
        X_window = window.drop('target', axis=1)
        y_window = window['target']
        
        for model_name, model in models.items():
            if model_name == 'AdaptiveRandomForest':
                # River model evaluation
                window_acc = []
                for idx, row in window.iterrows():
                    x = row.drop('target').to_dict()
                    y = row['target']
                    y_pred = model.predict_one(x)
                    if y_pred is not None:
                        window_acc.append(1.0 if y_pred == y else 0.0)
                    model.learn_one(x, y)
                if window_acc:
                    accuracy_rf.append(np.mean(window_acc))
            else:
                # SKLearn models evaluation
                y_pred = model.predict(X_window)
                acc = accuracy_score(y_window, y_pred)
                
                if model_name == 'SGDClassifier':
                    accuracy_sgd.append(acc)
                    model.partial_fit(X_window, y_window)
                elif model_name == 'GaussianNB':
                    accuracy_nb.append(acc)
                    model.partial_fit(X_window, y_window, classes=np.unique(data['target']))
    
    return accuracy_sgd, accuracy_nb, accuracy_rf

def plot_results(accuracy_sgd, accuracy_nb, accuracy_rf):
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_sgd, label="SGDClassifier", linestyle="-", color="blue")
    plt.plot(accuracy_nb, label="GaussianNB", linestyle="--", color="green")
    plt.plot(accuracy_rf, label="AdaptiveRandomForest", linestyle="-.", color="red")
    
    plt.axvline(x=len(accuracy_sgd)//2, color="black", linestyle="--", label="Concept Drift")
    plt.title("Model Accuracy Over Time with Concept Drift")
    plt.xlabel("Window Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Run experiment
    accuracy_sgd, accuracy_nb, accuracy_rf = run_experiment()
    
    # Plot results
    plot_results(accuracy_sgd, accuracy_nb, accuracy_rf)
