# DISORDERED FUNCTIONAL CLASS MAPPINGS
# Based on Table S1: Ontology terms for disordered functional classes
# 
# Note: DisProt uses GO terms for binding functions and IDPO terms for
# structural states and linkers. We map both to our 6 functional classes.

# For DisProt 2025_12
FUNCTIONAL_CLASS_MAPPING_99 = {
    # PROTEIN BINDING
    'GO:0005515': 'Protein_binding',  # Protein binding
    'GO:0002039': 'Protein_binding',  # P53 binding
    'GO:0017025': 'Protein_binding',  # TBP-class protein binding 
    'GO:0050692': 'Protein_binding',  # Beta-catenin binding # does not exist in datasets
    'GO:0005516': 'Protein_binding',  # Calmodulin binding 
    'GO:0001223': 'Protein_binding',  # Transcription coactivator binding
    'GO:0019838': 'Protein_binding',  # Growth factor binding  # does not exist in datasets
    'GO:0042393': 'Protein_binding',  # Histone binding  
    'GO:0017124': 'Protein_binding',  # SH3 domain binding 
    'GO:0031625': 'Protein_binding',  # Ubiquitin protein ligase binding
    'GO:0005161': 'Protein_binding',  # Platelet-derived growth factor receptor binding
    'GO:0019901': 'Protein_binding',  # Protein kinase binding
    'GO:0042288': 'Protein_binding',  # MHC class I protein binding
    'GO:0046982': 'Protein_binding',  # MDM2/MDM4 family protein binding (protein heterodimerization)
    'GO:0019904': 'Protein_binding',  # 14-3-3 protein binding
    'GO:0070063': 'Protein_binding',  # RNA polymerase binding 
    'GO:0061676': 'Protein_binding',  # Importin-alpha family protein binding
    # 'GO:0035035': 'Protein_binding',  # Histone acetyltransferase binding 
    # 'GO:0031491': 'Protein_binding',  # Nucleosome binding
    # 'GO:0030332': 'Protein_binding',  # Cyclin binding 
    # 'GO:0060090': 'Protein_binding',  # Molecular adaptor activity (262 occurrences - scaffolding)
    # 'GO:0042802': 'Protein_binding',  # Identical protein binding (homodimerization)
    # 'GO:0042803': 'Protein_binding',  # Protein homodimerization activity

    # NUCLEIC ACID BINDING (Combined DNA & RNA)
    # merged in create_dataset.py, kept separated here for parsing
    'GO:0003676': 'Nucleic_acid_binding',  # Nucleic acid binding 

    # DNA BINDING
    # mapped to Nucleic_acid_binding
    'GO:0003677': 'Nucleic_acid_binding',  # DNA binding
    'GO:0003697': 'Nucleic_acid_binding',  # Single-stranded DNA binding
    'GO:0008301': 'Nucleic_acid_binding',  # DNA binding, bending
    'GO:0051880': 'Nucleic_acid_binding',  # G-quadruplex DNA binding - does not exist in datasets

    # RNA BINDING
    # mapped to Nucleic_acid_binding
    'GO:0003723': 'Nucleic_acid_binding',  # RNA binding 
    'GO:0003729': 'Nucleic_acid_binding',  # mRNA binding
    'GO:0019843': 'Nucleic_acid_binding',  # rRNA binding
    'GO:0000049': 'Nucleic_acid_binding',  # tRNA binding
    'GO:0003727': 'Nucleic_acid_binding',  # Single-stranded RNA binding
    'GO:0003730': 'Nucleic_acid_binding',  # RNA stem-loop binding
    'GO:0001069': 'Nucleic_acid_binding',  # Regulatory region RNA binding
    'GO:0002151': 'Nucleic_acid_binding',  # G-quadruplex RNA binding

    # ION BINDING
    'GO:0043167': 'Ion_binding',  # Ion binding
    'GO:0046872': 'Ion_binding',  # Metal ion binding
    'GO:0005509': 'Ion_binding',  # Calcium ion binding
    'GO:0008270': 'Ion_binding',  # Zinc ion binding
    'GO:0046914': 'Ion_binding',  # Copper ion binding
    'GO:0030955': 'Ion_binding',  # Potassium ion binding
    'GO:0005506': 'Ion_binding',  # Iron ion binding

    # LIPID BINDING
    'GO:0008289': 'Lipid_binding',  # Lipid binding

    # FLEXIBLE LINKER
    'IDPO:0000033': 'Flexible_linker',  # Flexible linker/spacer (from IDPO)
}
