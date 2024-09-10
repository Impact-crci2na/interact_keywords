from dataclasses import dataclass, field
from typing import Optional, Dict
import networkx as nx
import numpy as np
import pickle
import word2vec
from goatools.obo_parser import GODag  # Assurez-vous que la bibliothèque goatools est installée
from gensim.models import Word2Vec
from goatools import obo_parser
from typing import Dict, Optional
import json
import bioservices
import requests
import seaborn as sns
import pandas as pd
from node2vec import Node2Vec
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import dill as pickle
from bioservices.uniprot import UniProt
u = UniProt(verbose = False)
import scipy.stats as stats
from collections import Counter
import time
from scipy.spatial import distance

def create_initial_graph(edges_file):
  G=nx.Graph()
  with open(edges_file,"r") as f:
      for l in f.readlines():
          for m in l.split():
              if m not in G.nodes():
                  G.add_node(m)
          G.add_edge(l.split()[0],l.split()[1])
  print(len(G.nodes()))
  print(len(G.edges()))
  return G

def list_keyword(list_node,dico_bp,go,kw):

    list_k=[]
    for i in list_node:
        if i in dico_bp.keys():
            match=False
            for v in dico_bp[i]:
                bp=go[v].name
                for k in kw:
                    if k in bp:
                        match=True
            if match==True and  i not in list_k:
                list_k.append(i)
    return list_k

def shortest_path(start_GN,list_prot,G):
    G1=nx.Graph()
    list_query=[]
    list_node=[]
    for i in list_prot:
        list_query.append(i[0])
        if i[0] not in G1.nodes():
            G1.add_node(i[0])
        if i[0] not in list_node:
            list_node.append(i[0])
        path=nx.shortest_path(G,source=start_GN,target=i[0])
        j=0
        while j<len(path)-1:
            name_p=path[j]
            name_p1=path[j+1]
            if name_p not in G1.nodes():
                G1.add_node(name_p)
            if path[j] not in list_node:
                list_node.append(path[j])
            if name_p1 not in G1.nodes():
                G1.add_node(name_p1)
            if path[j+1] not in list_node:
                list_node.append(path[j+1])
            G1.add_edge(name_p,name_p1)
            j+=1
    return list(G1.nodes()), G1 , list_query
    #prot_name='FAM111B'#EPB41L3
    # keyword=['ubiquitin']
def top_similar(list_k,model,prot_name,top):
    dico_score={}
    final=[]
    for i in list_k:
        score=model.wv.similarity(prot_name,i)
        dico_score[score]=i
    top_list=sorted(dico_score.keys(),reverse=True)
    for s in top_list[:top]:
        final.append((dico_score[s],round(s,3)))
    return final

def write_top_list(top_list,dico_bp,go):
  with open("liste_proteines.txt","w") as f:
    for i in top_list:
      list_bp=[]
      acc=convert_GENENAME_ACC(i)
      for b in dico_bp[i]:
        if go[b].name not in list_bp:
          list_bp.append(go[b].name)
      list_mf=[]
      r=requests.get(f"http://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={acc}")
      data=r.json()
      for j in data["results"]:
        if j["goAspect"]=="molecular_function" and j["goId"] not in list_mf:
            list_mf.append(go[j["goId"]].name)
      f.write(f"nom :{i}\nProcessus biologique:\n{list_bp}\nFonction moléculaire:\n{list_mf}\n\n")
def convert_GENENAME_ACC(id) :
    convert = u.mapping(fr="Gene_Name", to="UniProtKB", query=id)
    for value in range(len(convert)) :
        if convert['results'][value]['to']['organism']['taxonId'] == 9606 :
            print(convert['results'][value]['to']['primaryAccession'])
    return convert['results'][value]['to']['primaryAccession']


def convert(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def display_shortest_path(G,start_GN,list_prot,model, filename = "Graph_SP"):
    color_map=[]
    for node in G:
        if node==start_GN:
            color_map.append('red')
        elif node in list_prot:
            color_map.append("blue")
        else:
            color_map.append('green')
    list_size=[]
    for s in G.nodes():
        score=model.wv.similarity(start_GN,s)
        list_size.append(score)
    fig = plt.figure(1, figsize=(40, 20))
    pos = nx.spring_layout(G,scale=4)
    nx.draw(G,pos,with_labels=True,node_color=color_map,node_size=[v*15500 for v in list_size],bbox=dict(facecolor="white", edgecolor='black', boxstyle='round,pad=0.2'), font_size=25)
    plt.savefig(f"{filename}.png", format="png")
    print("saving")
    plt.close('all')


def display_shortest_path_special(G, start_GN, list_prot, model, weight_matrix,filename= "Graph_SP"):
    color_map = []
    for node in G:
        if node == start_GN:
            color_map.append('red')
        elif node in list_prot:
            color_map.append("blue")
        else:
            color_map.append('green')

    list_size = []
    for s in G.nodes():
        score = model.wv.similarity(start_GN, s)
        list_size.append(score)

    edge_weights = {}
    for s in G.nodes():
        if s != start_GN and s in weight_matrix.index:
            weight = weight_matrix.at[start_GN, s]
            edge_weights[(start_GN, s)] = weight

    fig = plt.figure(1, figsize=(40, 20))
    pos = nx.spring_layout(G, scale=4)

    nx.draw(G, pos, with_labels=True, node_color=color_map,
            node_size=[v * 15500 for v in list_size],
            bbox=dict(facecolor="white", edgecolor='black', boxstyle='round,pad=0.2'),
            font_size=25)
    print(edge_weights)
    # Dessiner les arêtes avec des poids spécifiques
    for (u, v,d) in G.edges(data=True):
        if (u == start_GN or v == start_GN) and (u, v) in edge_weights or (v, u) in edge_weights:
            try:
                weight = edge_weights[(u, v)]
            except:
                weight = edge_weights[(v,u)]
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width= np.exp(-weight), edge_color='black')

    plt.savefig(f"{filename}.png", format="png")
    plt.close('all')

def find_common_nodes(graph_list):
    if not graph_list:
        return set()

    # Initialiser l'ensemble des nœuds communs avec les nœuds du premier graphe
    common_nodes = set(graph_list[0].nodes())

    # Intersecter les nœuds de tous les graphes
    for G in graph_list[1:]:
        print(G)
        common_nodes.intersection_update(G.nodes())

    return common_nodes




@dataclass
class sub_graph:
    target: str  # cible protéique initiale
    keywords: list[str]
    graph: Optional[nx.Graph] = None
    vector: Optional[np.array] = None
    neigbhour: list[str] = field(default_factory=list)
    list_prot: list[str] = field(default_factory=list)
    top_list: list = field(default_factory=list)
    list_k: list[str] = field(default_factory=list)
    list_query: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    def get_vector(self):
        return(self.vector)
    def get_name(self):
        return(self.target)
    def initialize(self, string_graph, dico_bp, go_data, node2vec_model):
        self.list_k = list_keyword(list(string_graph.nodes()), dico_bp, go_data, self.keywords)
        print(len(self.list_k))
        if len(self.list_k) > 0:
            try:
                top_list = top_similar(self.list_k, node2vec_model, self.target, 30)
                print(f"top liste = {top_list}")
                self.list_prot, self.graph , self.list_query = shortest_path(self.target, top_list, string_graph)
                print(self.graph)
            except Exception as e:
                self.errors.append(str(e))
        else:
            self.errors.append("saisir un autre identifiant uniprot list = 0")

    def create_vector_of_graph(self):
        self.vector = generate_graph_embedding(self.graph)

def generate_graph_embedding(G, dimensions=32, walk_length=10, num_walks=200, window=5):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window, min_count=1, batch_words=4)
    # Aggregation of node embeddings to form a graph-level vector
    graph_embedding = np.mean([model.wv[str(node)] for node in G.nodes()], axis=0)
    return graph_embedding



@dataclass
class graph_gestor:
    target : str   # cible protéique intial
    keywords : list[str] #= field(default_factory=list) # keywords d'origine, obtneue en passant la fonction
    node2vec_model: Optional[Word2Vec] = field(init=False, default=None) # model node2vec déja pre-entrainer
    matrix : Optional[pd.DataFrame] =None #matrice de similairaité
    string_graph:Optional[nx.Graph] = None # contient le graph obtneue avevc le réseaux string
    go_data: Optional[GODag] = field(init=False, default=None)# fihcier go basic
    dico_bp : Dict = field(default_factory=dict)
    graph:Optional[nx.Graph] = None # contient le graph obtneue avevc la protéine
    vector : Optional[np.array] = None # vecteurr propre de graph
    list_of_graph: list = field(default_factory=list) #"liste contenant des instance de graphe "
    list_k: list[str] = field(default_factory=list)
    list_query : list = field(default_factory=list)
    neigbhour :list = field(default_factory=list)
    top_list: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    list_of_important : list = field(default_factory=list)

    def init(self, go_file = "go-basic.obo", model_file="embeddings.model",bp_file="bp_string2.json",initial_edges="edges_string.txt"):
        try:

            self.string_graph = create_initial_graph(initial_edges)
            self.go_data = obo_parser.GODag(go_file)
            self.node2vec_model = Word2Vec.load(model_file)
            print(self.node2vec_model)
            with open(bp_file,"r") as f:
                self.dico_bp=json.load(f)
            self.list_k=list_keyword(list(self.string_graph.nodes()),self.dico_bp,self.go_data,self.keywords)
        except Exception as e:
            print(e)


    def fit_to_data(self):
        print(self.list_k)
        print("taille de la liste: ",len(self.list_k))
        if len(self.list_k)>0:
            try:
                i =1
                print("toto")
                self.top_list=top_similar(self.list_k,self.node2vec_model,self.target,30)
                print("tata")
                self.list_prot , self.graph , self.list_query=shortest_path(self.target,self.top_list,self.string_graph)
                print("tutu")
                print(f"initial {self.list_query}")
                print(self.graph)
                print("titi")
                i=i+1
                display_shortest_path(self.graph , self.target , self.list_query, self.node2vec_model , filename = f"initial_graph{i}_{self.target}_{self.keywords[0]}")
                # write_top_list(self.list_prot,self.dico_bp,self.go_data)
            except Exception as e:
                print("*"*50)
                print("Cet identifiant n'est pas valide ou bien la protéine n'est pas présente dans l'intéractome")
                print("Veuillez saisir un autre identifiant UNIPROT")
                print("*"*50)
                print(e)

        else:
          print("*"*50)
          print("Aucun de ces mots clés n'est présent dans les processus biologiques")
          print("Veuillez saisir un autre mot clé")
          print("*"*50)


    def create_subgraph(self, all = False):
        if all == True :
            self.neigbhour = list(self.graph)
        else:
            self.neigbhour = list(self.graph.neighbors(self.target))
        for prot in self.neigbhour:
            protein_graph_gestor = sub_graph(target=prot, keywords=self.keywords)
            protein_graph_gestor.initialize(self.string_graph, self.dico_bp, self.go_data, self.node2vec_model)
            self.list_of_graph.append(protein_graph_gestor)
            print("*"*50)
            print(protein_graph_gestor.target)


    def create_vector_of_graph(self):
        print(self.target)
        self.vector = generate_graph_embedding(self.graph)
    def create_matrix_of_vector(self):
        dictof={}
        for protein in self.list_of_graph :
            protein.create_vector_of_graph()
            dictof[protein.get_name()] = protein.get_vector()
        dictof[self.target] = self.vector
        embeddings_df = pd.DataFrame.from_dict(dictof, orient='index')
        dist_matrix = distance.cdist(embeddings_df.values, embeddings_df.values, 'euclidean')
        self.matrix  = pd.DataFrame(dist_matrix, index=embeddings_df.index, columns=embeddings_df.index)
        print(self.matrix)
    def plot_matrix(self):
        graph_keys = (self.matrix.index)
        # Création de la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.matrix, annot=True, fmt=".2f", cmap='viridis', xticklabels=graph_keys, yticklabels=graph_keys)
        plt.title('Heatmap of Contextual Distance Matrix')
        plt.xlabel('Graphs')
        plt.ylabel('Graphs')
        plt.show()

    def create_graph(self):
        print("creation of the graph")
        print(self.list_prot)
        display_shortest_path(self.graph,self.target,self.list_query,self.node2vec_model, f"Graph_{self.target}")


    def create_special_graph(self):
        self.create_vector_of_graph()
        self.create_matrix_of_vector()
        display_shortest_path_special(self.graph,self.target,self.list_query,self.node2vec_model, self.matrix, f"graph_interactor_{self.target}_{self.keywords[0]}")
    def show_mds(self):
        mds = MDS(n_components=2, dissimilarity="precomputed")
        embeddings_2d = mds.fit_transform(self.matrix)  # Utiliser .values pour obtenir le numpy array

        # Visualisation
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', marker='o')

        # Utiliser les index du DataFrame pour l'annotation
        for i, txt in enumerate(self.matrix.index):
            plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

        plt.title('MDS Embedding of Proteins')
        plt.xlabel('MDS1')
        plt.ylabel('MDS2')
        plt.grid(True)
        plt.show()
    def run_over_stat(self,number=30):
        count = 0
        print("coucou")
        while count < number:
            print(count)
            self.create_vector_of_graph()
            self.create_matrix_of_vector()
            # self.create_special_graph()
            self.selections_with_thresh()
            count = count + 1
            pass


    def write_json(self):
        print("Taking informations")
        key = self.keywords[0]

        stat_data = self.stat_importance()
        data = {
             str(key):
             {
             self.target :
             {
                "list_query": self.list_query,
                "neighbour": self.neigbhour,
                "top_list": self.top_list,
                "statistiques" : stat_data
            }
            }
        }
        with open(f"Interact_keywords_{self.target}_{key}.json", 'w') as json_file:
            json.dump(data, json_file, indent=4, default=convert)
        print(f"Data saved to ")

    def save_graph(self):
        with open(f"graph_{self.target}_{self.keywords[0]}.pkl", 'wb') as f:
            pickle.dump(self.graph, f)

    def selections_with_thresh(self):
        print(self.matrix)
        target_row = self.matrix.loc[self.target]
        threshold = target_row[target_row != 0.0].mean()
        filtered_columns = target_row[target_row < threshold]
        column_names = filtered_columns.index.tolist()
        values = filtered_columns.values.tolist()
        self.list_of_important.append(column_names)
        print(f"Colonnes avec des valeurs inférieures au seuil:{threshold}")
        print("Noms des colonnes:", column_names)
        print("Valeurs correspondantes:", values)
    def stat_importance(self):
        # Afficher la liste des éléments importants
        print(self.list_of_important)

        # Aplatir la liste de listes pour obtenir une liste de tous les éléments
        flat_list = [item for sublist in self.list_of_important for item in sublist]

        # Exclure 'FAM111B' de la liste aplatie
        flat_list = [item for item in flat_list if item != self.target]

        # Compter la fréquence d'apparition de chaque élément
        frequency_counter = Counter(flat_list)
        print("Fréquences des éléments:", frequency_counter)

        # Nombre total d'éléments
        total_elements = len(flat_list)

        # Nombre unique d'éléments
        unique_elements = len(frequency_counter)

        # Fréquence attendue si les éléments étaient distribués uniformément
        expected_frequency = total_elements / unique_elements

        # Effectuer un test du chi carré pour chaque élément
        results = {}
        for element, observed_frequency in frequency_counter.items():
            # Fréquences observées et attendues pour cet élément
            observed = [observed_frequency, total_elements - observed_frequency]
            expected = [expected_frequency, total_elements - expected_frequency]

            # Test du chi carré
            chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)

            # Stocker les résultats
            results[element] = {
                'observed_frequency': observed_frequency,
                'expected_frequency': expected_frequency,
                'chi2': chi2,
                'p-value': p ,
                "important" : False
            }

        # Afficher les résultats
        for element, result in results.items():
            print(f"Élément: {element}")
            print(f"  Fréquence observée: {result['observed_frequency']}")
            print(f"  Fréquence attendue: {result['expected_frequency']}")
            print(f"  Chi2: {result['chi2']}")
            print(f"  p-value: {result['p-value']}")
            if result['p-value'] < 0.05:
                print("  -> Différence significative dans la fréquence")
                results[element]["important"] = True
            else:
                print("  -> Pas de différence significative dans la fréquence")
        return results
    def important_node(self):
        list=[]
        for graphe in self.list_of_graph :
            list.append(graphe.graph)
        # list.append(self.graph)
        important = find_common_nodes(list)
        print(important)
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
# start_time = time.time()
#
#
# ## exemple d'utilisation:
# tam = graph_gestor("KRAS",['gene expression'])
# tam.init()
# tam.fit_to_data()
#
# tam.create_subgraph()
# tam.run_over_stat(1)
# tam.write_json()
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Temps d'exécution : {execution_time} secondes")
# go-basic.obo
# embeddings.model
# bp_string2.json
# edges_string.txt
