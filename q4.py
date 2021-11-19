import time
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
import q1


def calculate_mode_time(function, arguments):
    elapsed_time_list = []
    for i in range(10):
        start = time.time()
        function(arguments)
        time_required = time.time() - start
        elapsed_time_list.append(round(time_required, 6))
    
    mode = max(set(elapsed_time_list), key = elapsed_time_list.count)
    if int(mode) == 0:
        return sum(elapsed_time_list)/len(elapsed_time_list)
    
    return mode


if __name__ == "__main__":

    datasets = q1.Datasets()

    # EXECUTION TIME
    print("EXECUTION TIME OF ALGORITHMS (seconds)")

    print("\nKarate Club Network -")
    print("Girvan Neuman:", calculate_mode_time(girvan_newman, datasets.karate_net))
    print("Modularity Based:", calculate_mode_time(greedy_modularity_communities, datasets.karate_net))
    print("Spectral Clustering:", 
        calculate_mode_time(
            SpectralClustering(
                n_clusters=2, 
                affinity='precomputed',
                random_state=42
            ).fit, 
            datasets.karate_network
        )
    )

    # DOLPHIN NETWORK
    print("\nDolphin Network -")
    print("Girvan Neuman:", calculate_mode_time(girvan_newman, datasets.dolphin_net))
    print("Modularity Based:", calculate_mode_time(greedy_modularity_communities, datasets.dolphin_net))
    print("Spectral Clustering:", 
        calculate_mode_time(
            SpectralClustering(
                n_clusters=2, 
                affinity='precomputed',
                random_state=42
            ).fit, 
            datasets.dolphin_network
        )
    )


    # JAZZ MUSICIANS' NETWORK
    print("\nJazz Musicians' Network -")
    print("Girvan Neuman:", calculate_mode_time(girvan_newman, datasets.jazz_net))
    print("Modularity Based:", calculate_mode_time(greedy_modularity_communities, datasets.jazz_net))
    print("Spectral Clustering:", 
        calculate_mode_time(
            SpectralClustering(
                n_clusters=2, 
                affinity='precomputed',
                random_state=42
            ).fit, 
            datasets.jazz_network
        )
    )