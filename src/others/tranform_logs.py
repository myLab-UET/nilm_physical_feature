import re
import csv

# Define the path to the log file
log_file_path = '/home/mrcong/Code/mylab-nilm-files/nilm-physical-features/results/models/VNDALE2/window_1800/5_comb/rf_tuning/2025-01-17_rf_tuning.log'

# Initialize a list to store extracted values
data = []

# Define regex patterns to extract values
pattern_n_estimators = re.compile(r'n_estimators=(\d+)')
pattern_max_depth = re.compile(r'max_depth=(\d+)')
pattern_validation_accuracy = re.compile(r'Validation accuracy: ([\d.]+)')
pattern_validation_f1_score = re.compile(r'F1-Score: ([\d.]+)')
pattern_training_accuracy = re.compile(r'Training accuracy: ([\d.]+)')
pattern_training_f1_score = re.compile(r'F1-Score: ([\d.]+)')

# Initialize variables to store extracted values
n_estimators = None
max_depth = None
validation_accuracy = None
validation_f1_score = None
training_accuracy = None
training_f1_score = None

# Read the log file
with open(log_file_path, 'r') as file:
    for line in file:
        if 'Training Random Forest' in line:
            n_estimators = pattern_n_estimators.search(line).group(1)
            max_depth = pattern_max_depth.search(line).group(1)
        elif 'Validation accuracy' in line:
            validation_accuracy = pattern_validation_accuracy.search(line).group(1)
            validation_f1_score = pattern_validation_f1_score.search(line).group(1)
        elif 'Training accuracy' in line:
            training_accuracy = pattern_training_accuracy.search(line).group(1)
            training_f1_score = pattern_training_f1_score.search(line).group(1)
            # Append the extracted values to the data list
            data.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'validation_accuracy': validation_accuracy,
                'validation_f1_score': validation_f1_score,
                'training_accuracy': training_accuracy,
                'training_f1_score': training_f1_score,
                "difference": float(training_accuracy) - float(validation_accuracy)
            })

# Write extracted values to a CSV file
with open('rf_tuning_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['n_estimators', 'max_depth', 'validation_accuracy', 'validation_f1_score', 'training_accuracy', 'training_f1_score', 'difference']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(data)