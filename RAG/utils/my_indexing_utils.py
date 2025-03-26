import os
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from langchain_core.documents import Document
import random
from langchain_chroma import Chroma


def read(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def make_file_list(path):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        print(filenames)
        for filename in filenames:
            print(filename)
            if filename.endswith('.txt'): 
                list_of_files[filename.split('.')[0]] = os.sep.join([dirpath, filename])
    return list_of_files

def sample_file_list_by_month(file_dict, n=1):
    grouped = defaultdict(list)

    for key, path in file_dict.items():
        try:
            month_key = key[:7]  
            grouped[month_key].append((key, path))
        except:
            continue  

   
    sampled = {}
    for month, files in grouped.items():
        selected = random.sample(files, min(n, len(files)))
        for key, path in selected:
            sampled[key] = path

    return sampled
    

def makeDocuments(path, sample_size,verbose):
    if verbose:
        print()
    return [
        Document(
            page_content=read(file_path=value),
            metadata={"source": key}
        )
        for key, value in sample_file_list_by_month(make_file_list(path), n=sample_size).items()
    ]

def visualiser_densite_documents_par_mois(fichiers_dict, palette="Blues"):
    data = defaultdict(lambda: defaultdict(int))

    for filename in fichiers_dict:
        try:
            parts = filename.split('-')
            if len(parts) < 3:
                continue
            year, month = parts[0], int(parts[1])
            data[year][month] += 1
        except:
            continue

    annees = sorted(data.keys())
    mois = list(range(1, 13))

    matrice = []
    max_val = 0
    for annee in annees:
        ligne = [data[annee].get(mois_num, 0) for mois_num in mois]
        max_val = max(max_val, *ligne)
        matrice.append(ligne)

    base_cmap = plt.get_cmap(palette, max_val) 
    colors = ['black'] + [base_cmap(i) for i in range(base_cmap.N)]  
    cmap = mcolors.ListedColormap(colors)
    bounds = list(range(0, max_val + 2))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Affichage
    fig, ax = plt.subplots(figsize=(10, len(annees)))
    cax = ax.imshow(matrice, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"])
    ax.set_yticks(range(len(annees)))
    ax.set_yticklabels(annees)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Années")
    ax.set_title("Densité de documents par mois et par année")

    cbar = plt.colorbar(cax)
    cbar.set_label("Nombre de documents")

    plt.savefig("densite_documents.pdf", format="pdf", dpi=300, bbox_inches="tight")


def save_documents(docs, output_dir,verbose):
    if os.path.exists((output_dir)):
        if verbose:
            print("Directory already exists")
    else:
        os.makedirs(output_dir, exist_ok=True)
        
        for i, doc in enumerate(docs):
            filename = f"doc_{doc.metadata['chunk_id']}.txt"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(doc.page_content)
        if verbose:
            print("Splitted sub-documents where saved in f{output_dir_abs}")

def make_db(persist_dir,verbose,embeddings):
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        if verbose:
            print("Vector store déjà présent, chargement...")
        vector_store = Chroma(
            collection_name="documents_collection",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
    else:
        if verbose:
            print("Création du vector store...")
        vector_store = Chroma(
            collection_name="documents_collection",
            embedding_function=embeddings,
            persist_directory=persist_dir)
    return vector_store


def split_docs(text_splitter, documents_list, verbose):
    all_splits = []
    for doc in documents_list:
        source = doc.metadata["source"]
        if verbose:
            print(f"Splitting → {source}")
        
        splits = text_splitter.split_documents([doc])
        
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = f"{source}_chunk_{i}"
            split.metadata["parent_source"] = source
        
        all_splits.extend(splits)
    if verbose:
        print(f"Split blog post into {len(all_splits)} sub-documents.")
    return all_splits