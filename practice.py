#project: concept drift
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from river.ensemble import AdaptiveRandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from river.drift import ADWIN, DDM
import matplotlib.pyplot as plt
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Parameters
N_SAMPLES = 10000
CHUNK_SIZE = 100
N_FEATURES = 20
N_CLASSES = 2
DRIFT_POINT = 5000
CACHE_SIZE = 500
MAX_WORKERS = 4

def generate_data_with_drift():
    """Generate synthetic data with concept drift"""
    X1, y1 = make_classification(
        n_samples=DRIFT_POINT, 
        n_features=N_FEATURES, 
        n_classes=N_CLASSES, 
        random_state=42
    )
    X2, y2 = make_classification(
        n_samples=N_SAMPLES - DRIFT_POINT, 
        n_features=N_FEATURES, 
        n_classes=N_CLASSES, 
        random_state=99,
        flip_y=0.5
    )
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(N_FEATURES)])
    data['target'] = y
    
    return data

def process_chunk(chunk_idx, chunk_data, models, drift_detectors):
    """Process a single data chunk"""
    results = {'accuracies': {}, 'drift_points': {}}
    for model_name in models.keys():
        results['drift_points'][model_name] = []
    
    X_chunk = chunk_data.drop('target', axis=1)
    y_chunk = chunk_data['target']
    
    for model_name, model in models.items():
        try:
            if model_name == 'AdaptiveRandomForest':
                # River model
                chunk_acc = []
                for idx, row in chunk_data.iterrows():
                    x = row.drop('target').to_dict()
                    y = row['target']
                    y_pred = model.predict_one(x)
                    if y_pred is not None:
                        chunk_acc.append(1.0 if y_pred == y else 0.0)
                    model.learn_one(x, y)
                acc = np.mean(chunk_acc) if chunk_acc else 0.0
            else:
                # Sklearn models
                if chunk_idx == 0:  # Initial fit
                    if model_name == 'SGD':
                        model.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
                    else:
                        model.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
                    acc = 0.0
                else:
                    y_pred = model.predict(X_chunk)
                    acc = accuracy_score(y_chunk, y_pred)
                    model.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1]))
        except Exception as e:
            print(f"Error in {model_name}: {str(e)}")
            acc = 0.0
        
        results['accuracies'][model_name] = acc
        
        # Drift detection
        for detector in drift_detectors:
            try:
                detector.update(1 - acc)
                if detector.change_detected:
                    results['drift_points'][model_name].append(
                        (detector.__class__.__name__, chunk_idx)
                    )
            except Exception as e:
                print(f"Error in drift detection for {model_name}: {str(e)}")
    
    return results

def run_experiment(models, drift_detectors):
    """Run the streaming experiment"""
    data = generate_data_with_drift()
    chunk_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunk = data.iloc[i:i+CHUNK_SIZE]
            futures.append(
                executor.submit(process_chunk, i//CHUNK_SIZE, chunk, models, drift_detectors)
            )
        
        for future in futures:
            try:
                result = future.result()
                chunk_results.append(result)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                chunk_results.append({
                    'accuracies': {name: 0.0 for name in models.keys()},
                    'drift_points': {name: [] for name in models.keys()}
                })
    
    return chunk_results

def plot_results(results):
    """Plot accuracy over time"""
    plt.figure(figsize=(12, 6))
    
    accuracies = {
        model_name: [chunk['accuracies'][model_name] for chunk in results]
        for model_name in models.keys()
    }
    
    colors = {'SGD': 'blue', 'NaiveBayes': 'green', 'AdaptiveRandomForest': 'red'}
    styles = {'SGD': '-', 'NaiveBayes': '--', 'AdaptiveRandomForest': '-.'}
    
    for model_name, accs in accuracies.items():
        plt.plot(accs, label=model_name, color=colors[model_name], linestyle=styles[model_name])
    
    plt.axvline(x=DRIFT_POINT//CHUNK_SIZE, color='black', linestyle='--', label='Drift Point')
    plt.title('Model Accuracy Over Time with Concept Drift')
    plt.xlabel('Chunk Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Initialize models
    models = {
        'SGD': SGDClassifier(random_state=42, loss='log_loss', warm_start=True),
        'NaiveBayes': GaussianNB(),
        'AdaptiveRandomForest': AdaptiveRandomForestClassifier(seed=42)
    }
    
    # Initialize drift detectors
    drift_detectors = [ADWIN(), DDM()]
    
    # Run experiment
    results = run_experiment(models, drift_detectors)
    
    # Plot results
    plot_results(results)
    
    # Calculate and save final results
    final_results = {}
    for model_name in models.keys():
        try:
            model_accuracies = [chunk['accuracies'][model_name] for chunk in results]
            model_accuracies = [acc for acc in model_accuracies if acc is not None]
            model_drifts = []
            for chunk in results:
                if chunk['drift_points'][model_name]:
                    model_drifts.extend(chunk['drift_points'][model_name])
            
            final_results[model_name] = {
                "average_accuracy": float(np.mean(model_accuracies)) if model_accuracies else 0.0,
                "drift_points": model_drifts
            }
        except Exception as e:
            print(f"Error processing results for {model_name}: {str(e)}")
            final_results[model_name] = {
                "average_accuracy": 0.0,
                "drift_points": []
            }
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=4)

