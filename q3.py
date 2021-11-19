from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
import q1

class Clusters:

    def __init__(self):
        self.datasets = q1.Datasets()


    def print_cluster_sizes(self, clusters):
        cluster_sizes = []
        for cluster in clusters:
            cluster_sizes.append(len(cluster))
        
        return cluster_sizes
    

    # CLUSTERS USING GIRVAN NEUMAN METHOD
    def cluster_girvan_neuman(self):
        self.gn_cluster_karate = list(sorted(cluster) for cluster in next(girvan_newman(self.datasets.karate_net)))
        self.gn_cluster_dolphin = list(sorted(cluster) for cluster in next(girvan_newman(self.datasets.dolphin_net)))
        self.gn_cluster_jazz = list(sorted(cluster) for cluster in next(girvan_newman(self.datasets.jazz_net)))


    # CLUSTERS USING MODULARITY BASED METHOD
    def clusters_modularity_based(self):
        self.mb_cluster_karate = greedy_modularity_communities(self.datasets.karate_net)
        self.mb_cluster_dolphin = greedy_modularity_communities(self.datasets.dolphin_net)
        self.mb_cluster_jazz = greedy_modularity_communities(self.datasets.jazz_net)


    # CLUSTERS USING SPECTRAL CLUSTERING METHOD USING THE GRAPH LAPLACIAN
    def cluster_spectral_laplacian(self):
        
        # KARATE CLUB
        sc_karate = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42).fit(self.datasets.karate_network)
        sc_clusters_map_karate = {}
        for i in range(len(sc_karate.labels_)):
            sc_clusters_map_karate[i] = sc_karate.labels_[i]
            
        cluster_0 = []
        cluster_1 = []
        for key, val in sc_clusters_map_karate.items():
            if val == 0:
                cluster_0.append(key)
            else:
                cluster_1.append(key)   
        self.sc_clusters_karate = [cluster_0, cluster_1]

        # DOLPHIN
        sc_dolphin = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42).fit(self.datasets.dolphin_network)
        sc_clusters_map_dolphin = {}
        for i in range(len(sc_dolphin.labels_)):
            sc_clusters_map_dolphin[i] = sc_dolphin.labels_[i]
            
        cluster_0 = []
        cluster_1 = []
        for key, val in sc_clusters_map_dolphin.items():
            if val == 0:
                cluster_0.append(key)
            else:
                cluster_1.append(key)    
        self.sc_clusters_dolphin = [cluster_0, cluster_1]

        # JAZZ MUSICIANS'
        sc_jazz = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42).fit(self.datasets.jazz_network)
        sc_clusters_map_jazz = {}
        for i in range(len(sc_jazz.labels_)):
            sc_clusters_map_jazz[i] = sc_jazz.labels_[i]
            
        cluster_0 = []
        cluster_1 = []
        for key, val in sc_clusters_map_jazz.items():
            if val == 0:
                cluster_0.append(key)
            else:
                cluster_1.append(key)    
        self.sc_clusters_jazz = [cluster_0, cluster_1]


if __name__ == "__main__":

    clusters = Clusters()

    clusters.cluster_girvan_neuman()
    clusters.clusters_modularity_based()
    clusters.cluster_spectral_laplacian()

    print("GIRVAN NEUMAN")
    print("\nNumber of clusters in graphs-")
    print("Karate Club Network:", len(clusters.gn_cluster_karate))
    print("Dolphin Network:", len(clusters.gn_cluster_dolphin))
    print("Jazz Musicians' Network:", len(clusters.gn_cluster_jazz))

    print("\nCluster sizes-")
    print("Karate Club Network:", clusters.print_cluster_sizes(clusters.gn_cluster_karate))
    print("Dolphin Network:", clusters.print_cluster_sizes(clusters.gn_cluster_dolphin))
    print("Jazz Musicians' Network:", clusters.print_cluster_sizes(clusters.gn_cluster_jazz))


    print("\n\nMODULARITY BASED")
    print("\nNumber of clusters in graphs-")
    print("Karate Club Network:", len(clusters.mb_cluster_karate))
    print("Dolphin Network:", len(clusters.mb_cluster_dolphin))
    print("Jazz Musicians' Network:", len(clusters.mb_cluster_jazz))

    print("\nCluster sizes-")
    print("Karate Club Network:", clusters.print_cluster_sizes(clusters.mb_cluster_karate))
    print("Dolphin Network:", clusters.print_cluster_sizes(clusters.mb_cluster_dolphin))
    print("Jazz Musicians' Network:", clusters.print_cluster_sizes(clusters.mb_cluster_jazz))


    print("\n\nSPECTRAL CLUSTERING")
    print("\nCluster sizes-")
    print("Karate Club Network:", clusters.print_cluster_sizes(clusters.sc_clusters_karate))
    print("Dolphins Network:", clusters.print_cluster_sizes(clusters.sc_clusters_dolphin))
    print("Jazz Musicians' Network:", clusters.print_cluster_sizes(clusters.sc_clusters_jazz))