import networkx as nx
from karateclub import DeepWalk, Node2Vec, Walklets
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
    # ? remove duplicates by keeping the first occurence
    df = df.drop_duplicates(subset='CodeElementId', keep='first')
    return df

def file_embedding_algo(node_mapping: dict, edge_list_file: list):
    """
    Calculate the embedding of the graph
    
    :param node_mapping: The mapping from node number to node name (e.g. {0: 'file1.java', 1: 'file2.java', ...})
    :param edge_list_file: The list of edges in the graph (e.g. 1->2, 2->3, ...)
    :return: embedding of the graph (matrix)
    """
        
    # create a graph
    G = nx.Graph()
        
    for key in node_mapping:
        G.add_node(key)
        
    # add edges to graph
    for edge in edge_list_file:
        try:
            G.add_edge(edge[0], edge[1])
        except ValueError:
            print('Error: node not found')
            
    model = DeepWalk()
    # model = Node2Vec()
    model.fit(G)

    embedding = model.get_embedding()
    
    return embedding

def file_mapping(project: str):
    """
    Calculate the embedding of the graph
    
    :param project: The name of the project
    :return: mapping of the nodes
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

    node_mapping = dict()
    for i, node in enumerate(nodes):
        node_mapping[i] = node
        
    edge_list_file_nr = []
    for edge in edge_list_file:
        edge_list_file_nr.append([list(nodes).index(edge[0]), list(nodes).index(edge[1])])
            
    
    return node_mapping, edge_list_file_nr

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
    # add all edges to the dictionary
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
    
    model = DeepWalk()
    # model = Node2Vec()
    # create a graph
    G = nx.Graph()
    
    
    for name in name_to_number_and_id:
        G.add_node(name_to_number_and_id[name]['count'])
        
        for id in name_to_number_and_id[name]['ids']: # for each type (e.g. class, interface, etc.)
            for connection in id_to_name_and_connections[id]['connections']: # for each connection
                
                # look up name that has this id
                for name in name_to_number_and_id:
                    if id in name_to_number_and_id[name]['ids']:
                        name_from = name
                        break
                
                # look up name that has this id
                for name in name_to_number_and_id:
                    if connection['to'] in name_to_number_and_id[name]['ids']:
                        name_to = name
                        break
                    
                
                # find the number of the name
                from_number = name_to_number_and_id[name_from]['count']
                to_number = name_to_number_and_id[name_to]['count']
                
                
                # add the connection to the graph
                if from_number != to_number: # ? no self loops
                    G.add_edge(from_number, to_number)
                
    # plt.figure(figsize=(12, 8))
    # nx.draw_networkx(G, with_labels=True)
    # plt.show()
    
    model.fit(G)
    embedding = model.get_embedding()
            
    
    return embedding, name_to_number_and_id

def random_pairs(file_embs: list, mod_embs: list):
    
    random_match = []
    
    for pos1, vector1 in enumerate(file_embs):
        random_match.append((pos1, np.random.randint(0, len(mod_embs))))
    
    return random_match

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

def files_groundTruth_overlap(project: str, files: dict):
    
    ground_truth = get_ground_truth(project)
    
    # remove all files from ground truth that do not end in .java
    ground_truth = ground_truth[ground_truth['CodeElementId'].str.endswith('.java')]
    
    matching_files = 0
    matching_files_list = []
    not_matching_files_list = []
    for key in files:
        if files[key] in ground_truth['CodeElementId'].tolist():
            matching_files_list.append(files[key]) # files that are in the implementation and in the ground truth
            matching_files += 1
            
        else:
            not_matching_files_list.append(files[key]) # files that are in the implementation but not in the ground truth
            
    not_matching_ground_truth = []
    for file in ground_truth['CodeElementId'].tolist():
        if file not in matching_files_list:
            not_matching_ground_truth.append(file) # files that are in the ground truth but not in the implementation
            
    # remove all files from ground truth that are not in the implementation
    ground_truth = ground_truth[ground_truth['CodeElementId'].isin(matching_files_list)]
    
    # remove all files from the implementation that are not in the ground truth
    files = {key: files[key] for key in files if files[key] in ground_truth['CodeElementId'].tolist()}
            
    return ground_truth, files

def calculate_accuracy_rmvd(pair_list: list, file_mapping: dict, model_mapping: dict, project: str):
    
    # ! calculate precision and recall on the files that are in the ground truth and in the implementation (100% possible)
    ground_truth, file_mapping = files_groundTruth_overlap(project, file_mapping)
    
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
        
        if pair[0] in file_mapping:
            implementation_file = file_mapping[pair[0]]
        
            # check if the implementation file is in the list of files
            for file in list_of_files:
                if file == implementation_file:
                    correct += 1
                    break
    
    print(f'\tCorrect (removed): {correct}')
    print(f'\tTotal files in implementation (removed): {len(file_mapping)}')
    print(f'\tTotal files in ground truth: {len(ground_truth)}')
    print(f'\tValid pairs ({round(len(file_mapping)/len(pair_list)*100, 2)}%): {len(file_mapping)}')
    print(f'\tPrecision (removed): {round(correct / len(file_mapping) * 100, 2)}%')
    print(f'\tRecall (removed): {round(correct / len(ground_truth) * 100, 2)}%\n')
    
    return correct / len(file_mapping),  correct / len(ground_truth)

def calculate_accuracy(pair_list: list, file_mapping: dict, model_mapping: dict, project: str):
    
    # ! calculate precision and recall on all files even if they are not even in both
    ground_truth = get_ground_truth(project)
    
    correct = 0
    # for each pair check if any of the 
    for pair in pair_list:
        # get all ids that map to this number
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
    
    print(f'\tCorrect: {correct}')
    print(f'\tTotal files in implementation: {len(pair_list)}')
    print(f'\tTotal files in ground truth: {len(ground_truth)}')
    print(f'\tValid pairs ({round(len(pair_list)/len(file_mapping)*100, 2)}%): {len(pair_list)}')
    print(f'\tPrecision: {round(correct / len(pair_list) * 100, 2)}%')
    print(f'\tRecall: {round(correct / len(ground_truth) * 100, 2)}%\n')
    
    return correct / len(pair_list), correct / len(ground_truth)


def evaluate_project(project: str, runs: int):
    
    sum_precision = 0
    sum_precision_rmvd = 0
    sum_recall = 0
    sum_recall_rmvd = 0
    
    # make the mapping between the file to a node number for the graph embedding
    file_map, file_edge_list = file_mapping(project)
    
    for i in range(runs):
        print(f'\tRun {i+1} --------------------------------------')
    
        # run ebmedding algorithm
        file_embedding = file_embedding_algo(file_map, file_edge_list)
        
        # model to graph embedding
        model_embedding, model_mapping = model_embedding_algo(project)
        
        # calculate cosine similarity
        pair_list = cosine_similarity(file_embedding, model_embedding)
        # pair_list = random_pairs(file_embedding, model_embedding)
        
        # calculate accuracy
        precision, recall = calculate_accuracy(pair_list, file_map, model_mapping, project)
        
        # calculate accuracy_v2
        precision_rmvd, recall_rmvd = calculate_accuracy_rmvd(pair_list, file_map, model_mapping, project)
        
        sum_precision += precision
        sum_precision_rmvd += precision_rmvd
        sum_recall += recall
        sum_recall_rmvd += recall_rmvd
    
    print('--------------------------------------')    
    print(f'\t\tAverage precision: {round((sum_precision / runs) * 100, 2)}%')
    print(f'\t\tAverage recall: {round((sum_recall / runs) * 100, 2)}%')
    print(f'\t\tAverage precision (removed): {round((sum_precision_rmvd / runs) * 100, 2)}%')
    print(f'\t\tAverage recall (removed): {round((sum_recall_rmvd / runs) * 100, 2)}%')
    
    
    return

if __name__ == '__main__':
    projects = ['bigbluebutton', 'jabref', 'mediastore', 'teammates', 'teastore']
    runs = 5
    for project in projects:
        print(f'Project: {project}')
        evaluate_project(project, runs)