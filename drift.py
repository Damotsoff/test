import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from river.ensemble import AdaptiveRandomForestClassifier
from sklearn.metrics import accuracy_score
from river.drift import ADWIN, DDM
import matplotlib.pyplot as plt
import json
from concurrent.futures import ThreadPoolExecutor

# Parameters
CHUNK_SIZE = 50
MAX_WORKERS = 4
DRIFT_POINT = 10  # Chunk index where drift begins
DRIFT_DURATION = 5  # Number of chunks over which drift occurs
DRIFT_STRENGTH = 0.5  # Strength of drift progression

def calculate_recovery_time_and_stability(model_accuracies, drift_point, drift_duration):
    """Calculate recovery time and stability after concept drift"""
    if len(model_accuracies) <= drift_point + drift_duration:
        return {
            "recovery_time": -1,
            "stability": np.inf
        }
    
    post_drift_accuracies = model_accuracies[drift_point + drift_duration:]
    pre_drift_accuracy = np.mean(model_accuracies[:drift_point]) if drift_point > 0 else 0
    
    recovery_time = None
    for i, acc in enumerate(post_drift_accuracies):
        if acc >= pre_drift_accuracy * 0.9:
            recovery_time = i
            break
    
    if recovery_time is not None:
        stable_accuracies = post_drift_accuracies[recovery_time:]
        stability = np.std(stable_accuracies) if stable_accuracies else 0
    else:
        stability = np.inf
    
    return {
        "recovery_time": recovery_time if recovery_time is not None else -1,
        "stability": float(stability)
    }

def load_and_preprocess_data():
    """Load and preprocess dataset"""
    data = pd.read_csv('german_data.csv')
    target_column = 'target'
    
    if data[target_column].dtype in ['float64', 'float32']:
        data[target_column] = (data[target_column] > data[target_column].median()).astype(int)
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    data_scaled = pd.concat([X_scaled, y], axis=1)
    
    return data_scaled, target_column

def introduce_gradual_drift(chunk_data, target_column, chunk_idx):
    """Simulate gradual concept drift"""
    drifted_chunk = chunk_data.copy()
    drift_progress = min(1, (chunk_idx - DRIFT_POINT + 1) / DRIFT_DURATION)
    
    flip_probability = drift_progress * DRIFT_STRENGTH
    noise_mask = np.random.random(len(drifted_chunk)) < flip_probability
    
    drifted_chunk.loc[noise_mask, target_column] = 1 - drifted_chunk.loc[noise_mask, target_column]
    
    return drifted_chunk

def process_chunk(chunk_idx, chunk_data, models, drift_detectors, target_column):
    """Process a single data chunk"""
    results = {'accuracies': {}, 'drift_points': {}}
    
    # Introduce gradual concept drift after DRIFT_POINT
    if chunk_idx >= DRIFT_POINT and chunk_idx < DRIFT_POINT + DRIFT_DURATION:
        chunk_data = introduce_gradual_drift(chunk_data, target_column, chunk_idx)
    
    X_chunk = chunk_data.drop(target_column, axis=1)
    y_chunk = chunk_data[target_column]
    
    for model_name, model in models.items():
        results['drift_points'][model_name] = []
        
        try:
            # Специальная обработка для разных моделей
            if model_name == 'AdaptiveRandomForest':
                chunk_acc = []
                for idx, row in chunk_data.iterrows():
                    x = row.drop(target_column).to_dict()
                    y = row[target_column]
                    y_pred = model.predict_one(x)
                    if y_pred is not None:
                        chunk_acc.append(1.0 if y_pred == y else 0.0)
                    model.learn_one(x, y)
                
                acc = np.mean(chunk_acc) if chunk_acc else 0.0
            else:
                # Для SGD и NaiveBayes
                if chunk_idx == 0:
                    # Первый чанк - обучение
                    model.fit(X_chunk, y_chunk)
                    acc = accuracy_score(y_chunk, model.predict(X_chunk))
                else:
                    # Предсказание и обновление
                    y_pred = model.predict(X_chunk)
                    acc = accuracy_score(y_chunk, y_pred)
                    
                    # Partial fit с обработкой ошибок
                    try:
                        model.partial_fit(X_chunk, y_chunk, classes=np.unique(y_chunk))
                    except Exception as partial_fit_error:
                        print(f"Partial fit error for {model_name}: {partial_fit_error}")
                        # Принудительная переобучение если partial_fit не сработал
                        model.fit(X_chunk, y_chunk)
        
        except Exception as e:
            print(f"Error in {model_name} at chunk {chunk_idx}: {str(e)}")
            acc = 0.0
        
        results['accuracies'][model_name] = acc
        
        # Drift detection
        for detector_idx, detector in enumerate(drift_detectors):
            try:
                detector.update(1 - acc)
                if detector.change_detected:
                    results['drift_points'][model_name].append({
                        'detector': type(detector).__name__,
                        'chunk_index': chunk_idx,
                        'accuracy': acc,
                        'detector_index': detector_idx
                    })
            except Exception as e:
                print(f"Error in drift detection for {model_name}: {str(e)}")
    
    return results

def run_experiment(models, drift_detectors):
    """Run the streaming experiment"""
    data, target_column = load_and_preprocess_data()
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    chunk_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunk = data.iloc[i:i+CHUNK_SIZE]
            futures.append(
                executor.submit(
                    process_chunk, 
                    i//CHUNK_SIZE, 
                    chunk, 
                    models, 
                    drift_detectors, 
                    target_column
                )
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

def plot_results(results, models):
    """Plot accuracy over time"""
    plt.figure(figsize=(12, 6))
    
    # Безопасное извлечение accuracies
    accuracies = {}
    for model_name in models.keys():
        model_accuracies = []
        for chunk in results:
            if 'accuracies' in chunk and model_name in chunk['accuracies']:
                model_accuracies.append(chunk['accuracies'][model_name])
            else:
                model_accuracies.append(0.0)
        accuracies[model_name] = model_accuracies
    
    colors = {'SGD': 'blue', 'NaiveBayes': 'green', 'AdaptiveRandomForest': 'red'}
    styles = {'SGD': '-', 'NaiveBayes': '--', 'AdaptiveRandomForest': '-.'}
    
    for model_name, accs in accuracies.items():
        plt.plot(accs, label=model_name, color=colors[model_name], linestyle=styles[model_name])
    
    # Highlight drift period
    plt.axvspan(DRIFT_POINT, DRIFT_POINT + DRIFT_DURATION, 
                color='red', alpha=0.2, label='Drift Period')
    
    plt.title(f'Model Accuracy Over Time with Drift\n'
              f'(Drift Start: {DRIFT_POINT}, Duration: {DRIFT_DURATION}, Strength: {DRIFT_STRENGTH})')
    plt.xlabel('Chunk Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Initialize models с улучшенной инициализацией
    models = {
        'SGD': SGDClassifier(
            random_state=42, 
            loss='log_loss', 
            warm_start=True, 
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=5  # Дополнительный параметр для стабильности
        ),
        'NaiveBayes': GaussianNB(),
        'AdaptiveRandomForest': AdaptiveRandomForestClassifier(seed=42)
    }
    
    # Initialize drift detectors
    drift_detectors = [
        ADWIN(delta=0.01),  # Более чувствительный
        DDM(min_num_instances=10)  # Корректные параметры
    ]
    
    # Run experiment
    results = run_experiment(models, drift_detectors)
    
    # Plot results
    plot_results(results, models)
    
    # Calculate and save final results
    final_results = {}
    for model_name in models.keys():
        try:
            model_accuracies = [
                chunk['accuracies'][model_name] 
                for chunk in results 
                if 'accuracies' in chunk and model_name in chunk['accuracies']
            ]
            
            # Улучшенное извлечение точек дрейфа
            model_drifts = []
            for chunk in results:
                if 'drift_points' in chunk and model_name in chunk['drift_points']:
                    model_drifts.extend(chunk['drift_points'][model_name])
            
            # Calculate recovery and stability metrics
            recovery_metrics = calculate_recovery_time_and_stability(
                model_accuracies, 
                DRIFT_POINT, 
                DRIFT_DURATION
            )
            
            final_results[model_name] = {
                "average_accuracy": {
                    "before_drift": float(np.mean(model_accuracies[:DRIFT_POINT]) if DRIFT_POINT > 0 else 0),
                    "during_drift": float(np.mean(model_accuracies[DRIFT_POINT:DRIFT_POINT+DRIFT_DURATION]) if model_accuracies[DRIFT_POINT:DRIFT_POINT+DRIFT_DURATION] else 0),
                    "after_drift": float(np.mean(model_accuracies[DRIFT_POINT+DRIFT_DURATION:]) if model_accuracies[DRIFT_POINT+DRIFT_DURATION:] else 0)
                },
                "drift_points": model_drifts,
                "recovery_metrics": {
                    "recovery_time": recovery_metrics['recovery_time'],
                    "stability": recovery_metrics['stability']
                }
            }
        except Exception as e:
            print(f"Error processing results for {model_name}: {str(e)}")
            final_results[model_name] = {
                "average_accuracy": {
                    "before_drift": 0.0,
                    "during_drift": 0.0,
                    "after_drift": 0.0
                },
                "drift_points": [],
                "recovery_metrics": {
                    "recovery_time": -1,
                    "stability": np.inf
                }
            }
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
if __name__ == "__main__":
    main()
