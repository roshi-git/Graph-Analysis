import os
from scipy.io import mmread
import networkx as nx


class Datasets:

    def __init__(self):
        # LOAD DATASETS
        self.cwd = os.getcwd()

        # KARATE CLUB NETWORK
        self.file_path_karate = os.path.join(self.cwd, 'dataset\\karate club\\karate.mtx')
        self.karate_network = mmread(self.file_path_karate).toarray()

        # DOLPHIN NETWORK
        self.file_path_dolphin = os.path.join(self.cwd, 'dataset\\soc-dolphins\\soc-dolphins.mtx')
        self.dolphin_network = mmread(self.file_path_dolphin).toarray()

        # JAZZ MUSICIAN NETWORK
        self.file_path_jazz = os.path.join(self.cwd, 'dataset\\jazz\\jazz.mtx')
        self.jazz_network = mmread(self.file_path_jazz).toarray()

        # CONVERT TO NETWORKX GRAPH
        self.karate_net = nx.from_numpy_array(self.karate_network)
        self.dolphin_net = nx.from_numpy_array(self.dolphin_network)
        self.jazz_net = nx.from_numpy_array(self.jazz_network)


if __name__ == "__main__":
    datasets = Datasets()
    print("Datasets loaded!")
    print("Dataset dimensions -")
    print("Karate network:", datasets.karate_network.shape)
    print("Dolphin network:", datasets.dolphin_network.shape)
    print("Jazz Musicians' network:", datasets.jazz_network.shape)