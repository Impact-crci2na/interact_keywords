from setuptools import setup, find_packages

setup(
    name="Interact_keywords",  # Remplace par le nom de ton package
    version="2.1",
    packages=find_packages(),  # Recherche automatiquement les packages dans ton répertoire
    install_requires=[
        "word2vec",  #pour l'utilisation du modèle
        "pyvis",         # Pour la visualisation des graphes
        "networkx",      # Pour la manipulation des graphes
        "matplotlib",    # Pour les graphiques et visualisations
        "requests",      # Pour faire des requêtes HTTP
        "bioservices",
        "dill",
        "node2vec",
        "word2vec"    # Pour accéder aux services web comme UniProt
    ],
    description="A package for network visualization creation of contextual network",
    author="Karen Sobriel, Grégoire Menard, Haladi Ayad",
    author_email="crcina.impact.bioinfo@gmail.com",
    url="https://gitlab.univ-nantes.fr/E179974Z/pie.git",  # Lien vers ton dépôt GitHub
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Version de Python minimum requise
)
