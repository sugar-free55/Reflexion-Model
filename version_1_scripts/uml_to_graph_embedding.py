import xml.etree.ElementTree as ET
import json
import networkx as nx
from karateclub import DeepWalk
import matplotlib.pyplot as plt


def read_uml(project: str, save_connections: bool, save_mapping: bool, save_embedding: bool, save_graph: bool):
    """
    Reads the UML file of the project and saves the connections between the elements and the mapping of the names to numbers.
    
    :param project: The name of the project
    :param save_connections: If the connections dictionary should be saved to a file
    :param save_mapping: If the mapping of names to numbers should be saved to a file
    :return: None
    """
    
    
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
    # make a graph from the connections
    ############################################
    E = nx.Graph()
    
    # make a list with all connections to make the graph (e.g. [(1, 3), (3, 4), ...])
    for name in name_to_number_and_id: # for each node
        for id in name_to_number_and_id[name]['ids']: # for each type (e.g. class, interface, etc.)
            for connection in id_to_name_and_connections[id]['connections']: # for each connection
                name_from = connection['from']
                name_to = connection['to']
                
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
                E.add_edge(from_number, to_number)
                
    
    ############################################
    # make the embedding
    ############################################
    
    model = DeepWalk()
    # create a graph
    G = nx.Graph()
    
    for name in name_to_number_and_id:
        G.add_node(name_to_number_and_id[name]['count'])
        
        for id in name_to_number_and_id[name]['ids']: # for each type (e.g. class, interface, etc.)
            for connection in id_to_name_and_connections[id]['connections']: # for each connection
                id_from = connection['from']
                id_to = connection['to']
                print(f'{id_from} -> {id_to}')
                
                # look up name that has this id
                for name in name_to_number_and_id:
                    if id in name_to_number_and_id[name]['ids']:
                        name_from = name
                        break
                
                for name in name_to_number_and_id:
                    if connection['to'] in name_to_number_and_id[name]['ids']:
                        name_to = name
                        break
                    
                print(f'{name_from} -> {name_to}')
                
                # find the number of the name
                from_number = name_to_number_and_id[name_from]['count']
                to_number = name_to_number_and_id[name_to]['count']
                
                print(f'{from_number} -> {to_number}')
                
                # add the connection to the graph
                G.add_edge(from_number, to_number)
        
    model.fit(G)
    embedding = model.get_embedding()
    

    ############################################
    # save all
    ############################################
    
    # safe connection to file
    if save_connections:
        with open(f'models/relations/{project}_relation_dict.json', 'w') as f:
            json.dump(id_to_name_and_connections, f)    
            
    # safe mapping to file
    if save_mapping:
        # safe the name to number dictionary to file
        with open(f'models/mapping/{project}_elements.json', 'w') as f:
            json.dump(name_to_number_and_id, f)
            
    if save_graph:
        plt.figure(figsize=(12, 8))
        nx.draw_networkx(E, with_labels=True)
        plt.savefig(f'models/graphs/{project}_graph.png', dpi=300)
     
    if save_embedding:       
        # save the embedding to a file
        with open(f'models/embedding/{project}_embedding.txt', 'w') as f:
            for item in embedding:
                f.write("%s\n" % item)
    
    ############################################
    # print overview
    ############################################
    
    # print overview of nodes and relations
    for name in name_to_number_and_id:
        # print(f'\t{name}')
        for id in name_to_number_and_id[name]['ids']:
            for connection in id_to_name_and_connections[id]['connections']:
                # print(f'\t\t{connection}')
                pass
    
    return

        
if __name__ == '__main__':
    projects = ['bigbluebutton', 'jabref', 'mediastore', 'teammates', 'teastore']
    for project in projects:
        print(f'Project: {project}')
        read_uml(
            project,
            save_connections=False,
            save_mapping=False,
            save_embedding=True,
            save_graph=False
        )
        

    
