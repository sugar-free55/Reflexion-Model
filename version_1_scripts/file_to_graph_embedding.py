import networkx as nx
from karateclub import DeepWalk
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def node_to_graphs(project):
    save_path = 'implementation/extracted_dependencies_graphs/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # read in the data
    df = pd.read_csv(f'implementation/extracted_dependencies_file/{project}_FileDependencies.csv')

    # only keep the last part of path (e.g. file.py)
    df['From File'] = df['From File'].str.split('/').str[-1]
    df['To File'] = df['To File'].str.split('/').str[-1]

    # only keep entries with .java at the end
    df = df[df['From File'].str.endswith('.java')]
    
    print(df.head()) # overview of the data

    # add nodes
    edge_list_file = df[['From File', 'To File']].values.tolist() 
    # todo: make index for each file and use that index instead of the file name as node

    E = nx.Graph()
    E.add_edges_from(edge_list_file)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(E, with_labels=True, font_size=5, node_size=5)
    # plt.savefig(f'{save_path}{project}_graph.png', dpi=300)

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
        
    # save the edges with file name to a file
    # with open(f'{save_path}/edges/{project}_edges.json', 'w') as f:
    #     json.dump(edge_list_file, f)

    # save the mapping to a file
    # with open(f'{save_path}/mapping/{project}_node_mapping.json', 'w') as f:
    #     json.dump(node_mapping, f)
            

    # add edges to graph
    number_edges = []
    for edge in edge_list_file:
        try:
            G.add_edge(list(nodes).index(edge[0]), list(nodes).index(edge[1]))
            number_edges.append((list(nodes).index(edge[0]), list(nodes).index(edge[1])))
        except ValueError:
            print('Error: node not found')


    model = DeepWalk()
    model.fit(G)

    embedding = model.get_embedding()
    
    with open(f'{save_path}/edges_number/{project}_edges.json', 'w') as f:
        json.dump(number_edges, f)

    # save the embedding to a file
    # with open(f'{save_path}/embedding/{project}_embedding.txt', 'w') as f:
    #     for item in embedding:
    #         f.write(f'{item}\n')
            
    return

if __name__ == '__main__':
    projects = ['bigbluebutton', 'jabref', 'mediastore', 'teammates', 'teastore']
    for project in projects:
        node_to_graphs(project)
