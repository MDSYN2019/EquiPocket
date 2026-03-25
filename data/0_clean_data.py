# coding=utf-8
import os
import sys
import json

def split_protein_ligand(complex_pdb,
                         protein_pdb=None,
                         ligand_pdb=None,
                         ligand_res_name=None):
    import MDAnalysis as mda

    universe = mda.Universe(complex_pdb)
    base_selection = "all"
    # Match previous behavior: solvent/hydrogen cleanup happened only when exporting protein.
    if protein_pdb:
        base_selection += " and not (resname HOH WAT SOL TIP3 TIP3P) and not name H*"

    filtered_atoms = universe.select_atoms(base_selection)

    if ligand_pdb:
        ligand_base_selection = "not (protein or nucleic)"
        if ligand_res_name is not None:
            count = 0
            for tmp_ligand_resname in ligand_res_name:
                ligand_atoms = filtered_atoms.select_atoms(
                    f"{ligand_base_selection} and resname {tmp_ligand_resname}"
                )
                if ligand_atoms.n_atoms == 0:
                    continue
                tmp_ligand_pdb = ligand_pdb.replace(".pdb", "") + str(count) + ".pdb"
                ligand_atoms.write(tmp_ligand_pdb)
                count += 1
        else:
            ligand_atoms = filtered_atoms.select_atoms(ligand_base_selection)
            if ligand_atoms.n_atoms > 0:
                ligand_atoms.write(ligand_pdb)

    # save protein
    if protein_pdb:
        receptor_atoms = filtered_atoms.select_atoms("protein or nucleic")
        if receptor_atoms.n_atoms > 0:
            receptor_atoms.write(protein_pdb)


def clean_pdb_0(input_file, output_file):
    import Bio
    from Bio.PDB import PDBParser, PDBIO, Select
    class NonHetSelect(Select):
        def accept_residue(self, residue):
            return 1 if Bio.PDB.Polypeptide.is_aa(residue,standard=True) else 0
    pdb = PDBParser().get_structure("protein", input_file)
    io = PDBIO()
    io.set_structure(pdb)
    io.save(output_file, NonHetSelect())

def clean_scPDB():
    all_use_PDB = json.load(open("0_original_data/all_use_PDB.json"))
    path_1 = "1_clean_data/scPDB"
    all_length = len(all_use_PDB)
    count = 0
    for uniprot in all_use_PDB:
        tmp_file_name = all_use_PDB[uniprot]
        count += 1
        tmp_path = os.path.join(path_1, tmp_file_name)
        os.makedirs(tmp_path, exist_ok=True)
        site_file_name = "0_original_data/scPDB/%s/site.mol2" % tmp_file_name
        ligand_file_name = "0_original_data/scPDB/%s/ligand.mol2" % tmp_file_name
        cavity_file_name = "0_original_data/scPDB/%s/cavity6.mol2" % tmp_file_name
        protein_file_name = "0_original_data/scPDB/%s/protein.mol2" % tmp_file_name
        out_prointe_file_name = os.path.join(tmp_path, "protein.pdb")
        out_ligand_file_name = os.path.join(tmp_path, "ligand.pdb")
        if os.path.exists(out_prointe_file_name):
            continue
        split_protein_ligand(protein_file_name, protein_pdb=out_prointe_file_name, ligand_pdb=None)
        split_protein_ligand(ligand_file_name, protein_pdb=None, ligand_pdb=out_ligand_file_name)

def clean_coach420():
    path_0 = "0_original_data/coach420"
    path_1 = "1_clean_data/coach420"
    all_length = len(os.listdir(path_0))
    count = 0
    dic_file_ligand_res = {}
    for line in open("0_original_data/coach420_milg"):
        if "CONFLICTS" in line:
            continue
        line = line.split()
        file_name = line[0].split("/")[-1]
        ligand_res_name = line[1].split(",")
        dic_file_ligand_res[file_name] = ligand_res_name
    for tmp_file_name in os.listdir(path_0):
        if tmp_file_name not in dic_file_ligand_res:
            continue
        ligand_res_name = dic_file_ligand_res[tmp_file_name]
        count += 1
        tmp_path_0 = os.path.join(path_0, tmp_file_name)
        tmp_path_1 = os.path.join(path_1, tmp_file_name.replace(".pdb", ""))
        os.makedirs(tmp_path_1, exist_ok=True)
        protein_file_name = os.path.join(tmp_path_0)
        out_prointe_file_name = os.path.join(tmp_path_1, "protein.pdb")
        out_ligand_file_name = os.path.join(tmp_path_1, "ligand.pdb")
        if os.path.exists(out_prointe_file_name):
            continue
        split_protein_ligand(protein_file_name,
                protein_pdb=out_prointe_file_name,
                ligand_pdb=out_ligand_file_name,
                ligand_res_name=ligand_res_name)
    exit()

def clean_holo4k():
    path_0 = "0_original_data/holo4k"
    path_1 = "1_clean_data/holo4k"
    all_length = len(os.listdir(path_0))
    count = 0
    dic_file_ligand_res = {}
    for line in open("0_original_data/holo4k_milg"):
        if "CONFLICTS" in line:
            continue
        line = line.split()
        file_name = line[0].split("/")[-1]
        ligand_res_name = line[1].split(",")
        dic_file_ligand_res[file_name] = ligand_res_name
    for tmp_file_name in os.listdir(path_0):
        if tmp_file_name not in dic_file_ligand_res:
            continue
        ligand_res_name = dic_file_ligand_res[tmp_file_name]
        count += 1
        tmp_path_0 = os.path.join(path_0, tmp_file_name)
        tmp_path_1 = os.path.join(path_1, tmp_file_name.replace(".pdb", ""))
        os.makedirs(tmp_path_1, exist_ok=True)
        protein_file_name = tmp_path_0
        out_prointe_file_name = os.path.join(tmp_path_1, "protein.pdb")
        out_ligand_file_name = os.path.join(tmp_path_1, "ligand.pdb")
        if os.path.exists(out_prointe_file_name):
            continue
        split_protein_ligand(protein_file_name,
                protein_pdb=out_prointe_file_name,
                ligand_pdb=out_ligand_file_name,
                ligand_res_name=ligand_res_name)


def clean_PDBbind():
    path_0 = "0_original_data/PDBbind/refined-set"
    path_1 = "1_clean_data/PDBbind"
    all_length = len(os.listdir(path_0))
    count = 0
    for tmp_file_name in os.listdir(path_0):
        if len(tmp_file_name) != 4:
            continue
        print(all_length, count, tmp_file_name)
        count += 1
        tmp_path_0 = os.path.join(path_0, tmp_file_name)
        tmp_path_1 = os.path.join(path_1, tmp_file_name)
        os.makedirs(tmp_path_1, exist_ok=True)
        protein_file_name = os.path.join(tmp_path_0, "%s_protein.pdb" % tmp_file_name)
        ligand_file_name = os.path.join(tmp_path_0, "%s_ligand.mol2" % tmp_file_name)
        out_prointe_file_name = os.path.join(tmp_path_1, "protein.pdb")
        out_ligand_file_name = os.path.join(tmp_path_1, "ligand.pdb")
        if os.path.exists(out_prointe_file_name):
            continue
        split_protein_ligand(protein_file_name, protein_pdb=out_prointe_file_name, ligand_pdb=None)
        split_protein_ligand(ligand_file_name, protein_pdb=None, ligand_pdb=out_ligand_file_name)

if __name__ == "__main__":
    print("clean scPDB")
    clean_scPDB()
    print("clean coach420")
    clean_coach420()
    print("clean holo4k")
    clean_holo4k()
    print("clean PDBbind")
    clean_PDBbind()
