import json
import numpy as np
import pandas as pd


def get_pairs(project: str):
    pairs = []
    with open(f'comparison/{project}_best_match.json', 'r') as file:
        pairs = json.load(file)
    return pairs

def get_model_mapping(project: str):
    with open(f'models/mapping/{project}_elements.json', 'r') as file:
        model_mapping = json.load(file)
    return model_mapping

def get_implementation_mapping(project: str):
    with open(f'implementation/extracted_dependencies_graphs/mapping/{project}_node_mapping.json', 'r') as file:
        implementation_mapping = json.load(file)
    return implementation_mapping

def get_ground_truth(project: str):
    path = f'ground truth/goldstandard-{project}.csv'
    with open(path, 'r') as file:
        df = pd.read_csv(file)
    df['CodeElementId'] = df['CodeElementId'].str.split('/').str[-1]
    return df
        
        
def calc_accuracy(project: str):
    pairs = get_pairs(project)
    model_mapping = get_model_mapping(project)
    file_mapping = get_implementation_mapping(project)
    ground_truth = get_ground_truth(project)
    
    correct = 0
    for pair in pairs:
        # get all id's that map to this number
        for key in model_mapping:
            if model_mapping[key]["count"] == pair[1]:
                model_ids = model_mapping[key]["ids"] # model_ids: list of ids
                list_of_files = []
                # add all CodeElementId of ground truth to list_of_files if ArchitectureElementId == model_ids
                for model_id in model_ids:
                    list_of_files.extend(ground_truth[ground_truth['ArchitectureElementId'] == model_id]['CodeElementId'].tolist())
                continue    
            
        implementation_file = file_mapping[str(pair[0])]
        
        for file in list_of_files:
            if file == implementation_file:
                correct += 1
                break
    
    print(f'Correct: {correct}')
    print(f'Total: {len(pairs)}')
    print(f'Accuracy: {round(correct / len(pairs) * 100, 2)}%\n')
        
        
        
        

if __name__ == '__main__':
    projects = ['bigbluebutton', 'jabref', 'mediastore', 'teammates', 'teastore']
    for project in projects:
        print(f'Project: {project}')
        calc_accuracy(project)