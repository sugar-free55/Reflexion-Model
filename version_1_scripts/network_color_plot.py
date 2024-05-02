import matplotlib.pyplot as plt
import numpy as np
import json
import networkx as nx
import pandas as pd
from itertools import count

def get_edges(project: str):
    edges = []
    path = f'implementation/extracted_dependencies_graphs/edges_number/{project}_edges.json'
    with open(path, 'r') as file:
        edges = json.load(file)
    return edges

def get_ground_truth(project: str):
    path = f'ground truth/goldstandard-{project}.csv'
    with open(path, 'r') as file:
        df = pd.read_csv(file)
    df['CodeElementId'] = df['CodeElementId'].str.split('/').str[-1]
    # only keep entries with .java at the end
    df = df[df['CodeElementId'].str.endswith('.java')]
    # remove duplicates by keeping the first occurence
    df = df.drop_duplicates(subset='CodeElementId', keep='first')
    return df

def get_implementation_mapping(project: str):
    with open(f'implementation/extracted_dependencies_graphs/mapping/{project}_node_mapping.json', 'r') as file:
        implementation_mapping = json.load(file)
    return implementation_mapping

def get_model_mapping(project: str):
    with open(f'models/mapping/{project}_elements.json', 'r') as file:
        model_mapping = json.load(file)
    return model_mapping

def group_plot(project: str):
    edges = get_edges(project)
    ground_truth = get_ground_truth(project)
    implementation_mapping = get_implementation_mapping(project)
    model_mapping = get_model_mapping(project)

    G = nx.Graph()

    modelID_to_colorNr = dict()
    color_count = 0
    # add edges to graph
    for edge in edges:
        # make dict with architecture element id to color number
        for node in edge:
            # print(node, implementation_mapping[str(node)])
            if implementation_mapping[str(node)] in ground_truth['CodeElementId'].values:
                archID = ground_truth[ground_truth['CodeElementId'] == implementation_mapping[str(node)]]['ArchitectureElementId'].values[0]
                
                if archID not in modelID_to_colorNr:
                    modelID_to_colorNr[archID] = color_count
                    color_count += 1
        try:
            G.add_edge(edge[0], edge[1])
        except ValueError:
            print('Error: node not found')
            
    color_map = []
    for node in G.nodes:
        # append colormap the color of the specific node
        node_name = implementation_mapping[str(node)]
        if node_name in ground_truth['CodeElementId'].values:
            archID = ground_truth[ground_truth['CodeElementId'] == node_name]['ArchitectureElementId'].values[0]
            color = plt.cm.tab20(modelID_to_colorNr[archID])
        else: 
            color = 'grey'
        
        color_map.append(color)
    print(len(color_map))
    print(len(G.nodes))
            
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(G, with_labels=False, font_size=5, node_size=50, node_color=color_map, )
    plt.title(f'{project} - Color grouped by UML node\n(Grey nodes are not in the ground truth)\nEdges are extracted dependencies')
    plt.show()
    return

if __name__ == '__main__':
    projects = ['bigbluebutton', 'jabref', 'mediastore', 'teammates', 'teastore']
    # for project in projects:
    #     group_plot(project)
    
    group_plot('mediastore')