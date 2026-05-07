import mne
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print(f"MNE version: {mne.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NetworkX version: {nx.__version__}")
print("\nAll libraries imported successfully!")

# Quick sanity check
G = nx.watts_strogatz_graph(23, 4, 0.1)
print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print("Setup complete — ready to start thesis work!")