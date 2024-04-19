
import re
import networkx as nx
import matplotlib.pyplot as plt

# File path, replace with your file path
file_path = 'models/uml/teastore.uml'
# initialization chart
G = nx.DiGraph()

# Read file content
with open(file_path, 'r') as file:
    content = file.read()

# Use regular expressions to find all nodes of interfaces and components
element_pattern = r'<packagedElement xmi:type="uml:(Interface|Component)" xmi:id="([^"]+)" name="([^"]+)"'
elements = re.findall(element_pattern, content)

# Create a node mapping, where a name may correspond to multiple IDs
name_to_ids = {}
for _, element_id, element_name in elements:
    if element_name not in name_to_ids:
        name_to_ids[element_name] = []
    name_to_ids[element_name].append(element_id)

# Add a node for each name, using the name as the identifier for the node
for name in name_to_ids:
    G.add_node(name, label=name)

# Find all edges using regular expressions
edge_pattern = r'<(interfaceRealization|packagedElement xmi:type="uml:Usage") xmi:id="[^"]+" client="([^"]+)" supplier="([^"]+)"'
edges = re.findall(edge_pattern, content)

# Adding Edges to a Diagram
for _, client_id, supplier_id in edges:
    # Finding client and supplier names
    client_name = next((name for name, ids in name_to_ids.items() if client_id in ids), None)
    supplier_name = next((name for name, ids in name_to_ids.items() if supplier_id in ids), None)

    # Add edges using names as node identifiers
    if client_name and supplier_name:
        G.add_edge(client_name, supplier_name)

# visualization chart
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx(G, pos, with_labels=True, labels=labels, node_size=2000, font_size=10, font_weight='bold', node_color='skyblue',
        edge_color='k')
plt.show()