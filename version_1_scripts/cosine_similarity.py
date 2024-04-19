import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import json



def txt_to_list(project: str):
    """
    read the node mapping from a file and return it as a list
    
    :param project: The name of the project
    :return: The list of nodes
    """
    
    implementation_embedding = f'implementation/extracted_dependencies_graphs/embedding/{project}_embedding.txt'
    model_embedding = f'models/embedding/{project}_embedding.txt'
    
    implementation_embedding_matrix = []
    with open(implementation_embedding, 'r') as file:
        current_list = []
        for line in file:
            # Remove leading and trailing whitespace
            line = line.strip()
            # Check if the line starts with an opening square bracket
            if line.startswith('['):
                # If it does, initialize a new list
                current_list = []
                # Remove the opening square bracket
                line = line[1:]
            # Check if the line ends with a closing square bracket
            if line.endswith(']'):
                # If it does, remove the closing square bracket
                line = line[:-1]
                # Split the line by whitespace and convert each element to float
                current_list.extend([float(item) for item in line.split()])
                # Append the list to the lists
                implementation_embedding_matrix.append(current_list)
            else:
                # Split the line by whitespace and convert each element to float
                current_list.extend([float(item) for item in line.split()])

    model_embedding_matrix = []
    with open(model_embedding, 'r') as file:
        current_list = []
        for line in file:
            # Remove leading and trailing whitespace
            line = line.strip()
            # Check if the line starts with an opening square bracket
            if line.startswith('['):
                # If it does, initialize a new list
                current_list = []
                # Remove the opening square bracket
                line = line[1:]
            # Check if the line ends with a closing square bracket
            if line.endswith(']'):
                # If it does, remove the closing square bracket
                line = line[:-1]
                # Split the line by whitespace and convert each element to float
                current_list.extend([float(item) for item in line.split()])
                # Append the list to the lists
                model_embedding_matrix.append(current_list)
            else:
                # Split the line by whitespace and convert each element to float
                current_list.extend([float(item) for item in line.split()])
    
    return implementation_embedding_matrix, model_embedding_matrix

def cosine_similarity(project: str):
    
    # read the embeddings
    imp_embs, mod_embs = txt_to_list(project)
    
    best_match = []
    for pos1, vector1 in enumerate(imp_embs):
        current_best = -100
        best_pair = []
        for pos2, vector2 in enumerate(mod_embs):
            # calculate the cosine similarity
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
            
            # check if the current cosine similarity is better than the current best
            if cosine_similarity > current_best:
                current_best = cosine_similarity
                best_pair = [pos1, pos2]
            
        best_match.append(tuple(best_pair))   
        
    for pair in best_match:
        print(f'\t{pair}')
        
    # save best match to json file
    save_path = f'comparison'
    with open(f'{save_path}/{project}_best_match.json', 'w') as file:
        json.dump(best_match, file)
    
    return


if __name__ == '__main__':
    projects = ['bigbluebutton', 'jabref', 'mediastore', 'teammates', 'teastore']
    for project in projects:
        print(f'Project: {project}')
        cosine_similarity(project)
    