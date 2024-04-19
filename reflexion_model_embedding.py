import networkx as nx
from karateclub import DeepWalk
from karateclub import Node2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import xml.etree.ElementTree as ET

def get_ground_truth(project: str):
    path = f'ground truth/goldstandard-{project}.csv'
    with open(path, 'r') as file:
        df = pd.read_csv(file)
    df['CodeElementId'] = df['CodeElementId'].str.split('/').str[-1]
    return df

def file_embedding_algo(project: str):
    """
    Calculate the embedding of the graph
    
    :param project: The name of the project
    :return: The embedding of the graph, the mapping of the nodes
    """
    
    # read in the file structure
    df = pd.read_csv(f'implementation/extracted_dependencies_file/{project}_FileDependencies.csv')
    
    # only keep the last part of path (e.g. file.py)
    df['From File'] = df['From File'].str.split('/').str[-1]
    df['To File'] = df['To File'].str.split('/').str[-1]
    
    # only keep entries with .java at the end
    df = df[df['From File'].str.endswith('.java')]
    
    # add nodes
    edge_list_file = df[['From File', 'To File']].values.tolist() 
    
    # store elements in a set to avoid duplicates
    nodes = set()
    for edge in edge_list_file:
        nodes.add(edge[0])
        nodes.add(edge[1])
        
    # create a graph
    G = nx.Graph()

    node_mapping = dict()
    for i, node in enumerate(nodes):
        G.add_node(i)
        node_mapping[i] = node
        
    # add edges to graph
    for edge in edge_list_file:
        try:
            G.add_edge(list(nodes).index(edge[0]), list(nodes).index(edge[1]))
        except ValueError:
            print('Error: node not found')
            
    model = DeepWalk()
    model.fit(G)

    embedding = model.get_embedding()
    
    return embedding, node_mapping

def model_embedding_algo(project: str):
    
    # Parse the XML
    tree = ET.parse(f'models/uml/{project}.uml')
    root = tree.getroot()
    
    ############################################
    # create the dictionarys
    ############################################
    
    count = 0
    id_to_name_and_connections = {} # {id: {name: name, connections: [{from: id, to: id}, ...]}
    name_to_number_and_id = {} # {name: {count: number, ids: [id, ...]}}
    for node in root.iter('packagedElement'):
        # make entry for each id and save its name to the dictionary with list for connections
        if 'name' in node.attrib: # check if it is component and not a connection
            
            # create a dictionary with the name as key and a number and id as value to use for DeepWalk
            if node.attrib['name'][0] == 'I' and node.attrib['name'][0].isupper() and node.attrib['name'][1].isupper():
                if node.attrib['name'][1:] not in name_to_number_and_id:
                    name_to_number_and_id[node.attrib['name'][1:]] = {
                        'count': count,
                        'ids': [node.attrib['{http://www.omg.org/spec/XMI/20131001}id']]
                        }
                    count += 1
                    
            elif node.attrib['name'] not in name_to_number_and_id:
                name_to_number_and_id[node.attrib['name']] = {
                    'count': count,
                    'ids': [node.attrib['{http://www.omg.org/spec/XMI/20131001}id']]
                    }
                count += 1 
                
            else:
                name_to_number_and_id[node.attrib['name']]['ids'].append(node.attrib['{http://www.omg.org/spec/XMI/20131001}id'])

            # create a dictionary with the id as key and the name and connections as value
            id_to_name_and_connections[node.attrib['{http://www.omg.org/spec/XMI/20131001}id']] = {
                'name': node.attrib['name'],
                'connections': []
                }
            
    ############################################
    # add all  connections to the dictionary
    ############################################
    
    # add all element connections to the id's
    for node in root.iter('packagedElement'):
        if 'name' not in node.attrib:
            id_to_name_and_connections[node.attrib['client']]['connections'].append({
                    'from': node.attrib['client'],
                    'to': node.attrib['supplier']
                    }
                )
            
    # add all interface connections to the dictionary
    for node in root.iter('interfaceRealization'):
        id_to_name_and_connections[node.attrib['client']]['connections'].append({
                'from': node.attrib['client'], 
                'to': node.attrib['supplier']
                }
            )
        
    ############################################
    # make the embedding
    ############################################
    
    # model = DeepWalk()
    model = Node2Vec()
    # create a graph
    G = nx.Graph()
    
    for name in name_to_number_and_id:
        G.add_node(name_to_number_and_id[name]['count'])
        
        for id in name_to_number_and_id[name]['ids']: # for each type (e.g. class, interface, etc.)
            for connection in id_to_name_and_connections[id]['connections']: # for each connection
                id_from = connection['from']
                id_to = connection['to']
                
                # look up name that has this id
                for name in name_to_number_and_id:
                    if id in name_to_number_and_id[name]['ids']:
                        name_from = name
                        break
                
                for name in name_to_number_and_id:
                    if connection['to'] in name_to_number_and_id[name]['ids']:
                        name_to = name
                        break
                    
                
                # find the number of the name
                from_number = name_to_number_and_id[name_from]['count']
                to_number = name_to_number_and_id[name_to]['count']
                
                
                # add the connection to the graph
                G.add_edge(from_number, to_number)
        
    model.fit(G)
    embedding = model.get_embedding()
            
    
    return embedding, name_to_number_and_id

def cosine_similarity(file_embs: list, mod_embs: list):
    """
    Calculate the cosine similarity between the file embeddings and the model embeddings
    
    :param file_embs: The file embeddings
    :param mod_embs: The model embeddings
    :return: The best match between the file embeddings and the model embeddings
    """

    best_match = []
    for pos1, vector1 in enumerate(file_embs):
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
        
    return best_match

def calculate_accuracy(pair_list: list, file_mapping: dict, model_mapping: dict, project: str):
    
    ground_truth = get_ground_truth(project)
    
    correct = 0
    for pair in pair_list:
        # get all id's that map to this number
        for key in model_mapping:
            if model_mapping[key]["count"] == pair[1]:
                model_ids = model_mapping[key]["ids"] # model_ids: list of ids
                list_of_files = []
                # add all CodeElementId of ground truth to list_of_files if ArchitectureElementId == model_ids
                for model_id in model_ids:
                    list_of_files.extend(ground_truth[ground_truth['ArchitectureElementId'] == model_id]['CodeElementId'].tolist())
                continue    
            
        implementation_file = file_mapping[pair[0]]
        
        for file in list_of_files:
            if file == implementation_file:
                correct += 1
                break
    
    # print(f'\tCorrect: {correct}')
    # print(f'\tTotal: {len(pair_list)}')
    print(f'\t\tCurrent Precision: {round(correct / len(pair_list) * 100, 2)}%\n')
    
    # calc recall with how many pairs in ground truth are correctly predicted
    
    
    return correct / len(pair_list)

def evaluate_project(project: str, runs: int):
    
    sum_accuracy = 0
    for i in range(runs):
    
        # file to graph embedding
        file_embedding, file_mapping = file_embedding_algo(project)
        
        # model to graph embedding
        model_embedding, model_mapping = model_embedding_algo(project)
        
        # calculate cosine similarity
        pair_list = cosine_similarity(file_embedding, model_embedding)
        
        # calculate accuracy
        accurcacy = calculate_accuracy(pair_list, file_mapping, model_mapping, project)
        
        sum_accuracy += accurcacy
        
    print(f'\tAverage accuracy: {round((sum_accuracy / runs) * 100, 2)}%')
    
    
    return

if __name__ == '__main__':
    projects = ['bigbluebutton', 'jabref', 'mediastore', 'teammates', 'teastore']
    runs = 5 # ? this does not affect the result the way i expect it to
    for project in projects:
        print(f'Project: {project}')
        evaluate_project(project, runs)