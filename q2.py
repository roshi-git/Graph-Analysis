import numpy as np
import networkx as nx
import q1


if __name__ == "__main__":
    
    datasets = q1.Datasets()

    # GET NUMBER OF NODES OF GRAPHS
    print("Number of nodes-")
    print("Karate Club Network:", nx.number_of_nodes(datasets.karate_net))
    print("Dolphin Network:", nx.number_of_nodes(datasets.dolphin_net))
    print("Jazz Musicians' Network:", nx.number_of_nodes(datasets.jazz_net))

    # GET NUMBER OF EDGES OF GRAPHS
    print("\nNumber of edges-")
    print("Karate Club Network:", nx.number_of_edges(datasets.karate_net))
    print("Dolphin Network:", nx.number_of_edges(datasets.dolphin_net))
    print("Jazz Musicians' Network:", nx.number_of_edges(datasets.jazz_net))

    # GET AVERAGE SHORTEST PATH LENGTHS
    print("\nAverage shortest path lengths-")
    print("Karate Club Network:", round(nx.average_shortest_path_length(datasets.karate_net), 2))
    print("Dolphin Network:", round(nx.average_shortest_path_length(datasets.dolphin_net), 2))
    print("Jazz Musicians' Network:", round(nx.average_shortest_path_length(datasets.jazz_net), 2))

    # AVERAGE CLUSTERING COEFFICIENT
    print("\nAverage clustering coefficient-")
    print("Karate Club Network:", round(nx.average_clustering(datasets.karate_net), 2))
    print("Dolphin Network:", round(nx.average_clustering(datasets.dolphin_net), 2))
    print("Jazz Musicians' Network:", round(nx.average_clustering(datasets.jazz_net), 2))