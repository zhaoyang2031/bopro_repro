"""
Shared GA crossover/mutation module for offline MBO.

Adapted from GraphGA (Saturn) and GeneticGFN's GA operators.
In offline setting:
  1. Select top molecules from offline dataset
  2. Apply crossover + mutation
  3. Score new molecules with proxy model (no oracle calls during Phase 1)
  4. Return new SMILES + proxy scores for training
"""
import random
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

rdBase.DisableLog('rdApp.*')


# ============ Crossover operators (from GraphGA) ============

def cut(mol):
    """Cut a non-ring bond."""
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
        return None
    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])
    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except:
        return None


def cut_ring(mol):
    """Cut a ring bond."""
    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')))
            bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')))
            bis = ((bis[0], bis[1]), (bis[1], bis[2]),)
        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])
        try:
            frags = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(frags) == 2:
                return frags
        except:
            continue
    return None


def crossover(mol1, mol2):
    """Crossover two molecules."""
    if random.random() < 0.5:
        frags1 = cut(mol1)
        frags2 = cut(mol2)
    else:
        frags1 = cut_ring(mol1)
        frags2 = cut_ring(mol2)

    if frags1 is None or frags2 is None:
        return None
    if len(frags1) != 2 or len(frags2) != 2:
        return None

    # Combine fragments
    try:
        combo = Chem.CombineMol(frags1[0], frags2[1])
        Chem.SanitizeMol(combo)
        if combo.GetNumAtoms() >= 15 and combo.GetNumAtoms() <= 40:
            return combo
    except:
        pass
    return None


# ============ Mutation operators ============

SMARTS_MUTATIONS = [
    '[*:1]-[anium]>>[*:1]-[C][anium]',  # insert atom
    '[*:1]-[anium]>>[*:1]-[C]-[anium]',  # append atom
    '[#6]>>[#7]',  # change atom C→N
    '[#6]>>[#8]',  # change atom C→O
    '[#6]>>[#16]',  # change atom C→S
    '[#7]>>[#6]',  # change atom N→C
    '[#8]>>[#7]',  # change atom O→N
    '[*]=[*]>>[*]-[*]',  # change double to single bond
    '[*]-[*]>>[*]=[*]',  # change single to double bond
]


def mutate(mol, mutation_rate=0.1):
    """Apply random mutation to a molecule."""
    if random.random() > mutation_rate:
        return mol

    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol)
    # Simple mutation: random atom type change
    atoms = mol.GetAtoms()
    if len(atoms) == 0:
        return mol

    idx = random.randint(0, len(atoms) - 1)
    atom = atoms[idx]

    # Store original
    original_num = atom.GetAtomicNum()

    # Try different atom types
    new_types = [6, 7, 8, 16, 9, 17, 35]  # C, N, O, S, F, Cl, Br
    random.shuffle(new_types)
    for new_num in new_types:
        if new_num != original_num:
            atom.SetAtomicNum(new_num)
            try:
                Chem.SanitizeMol(mol)
                if 15 <= mol.GetNumAtoms() <= 40:
                    return mol
            except:
                atom.SetAtomicNum(original_num)
                continue

    return mol


# ============ Main GA function ============

def ga_crossover(smiles_list, scores_list, num_new=50, rank_coefficient=0.01):
    """
    Generate new molecules via GA crossover from top offline molecules.

    Args:
        smiles_list: List of SMILES strings (offline dataset)
        scores_list: List of scores (weighted sum of objectives)
        num_new: Number of new molecules to generate
        rank_coefficient: Rank-based sampling coefficient

    Returns:
        new_smiles: List of new SMILES strings
        new_scores: List of proxy scores (placeholder, to be filled by caller)
    """
    # Convert to RDKit Mols
    mols = []
    valid_scores = []
    for smi, score in zip(smiles_list, scores_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_scores.append(score)

    if len(mols) < 2:
        return [], []

    # Rank-based selection of parents
    scores_np = np.array(valid_scores)
    ranks = np.argsort(np.argsort(-1 * scores_np))
    weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)

    new_mols = []
    attempts = 0
    max_attempts = num_new * 20

    while len(new_mols) < num_new and attempts < max_attempts:
        # Select two parents
        parent_indices = list(np.random.choice(
            len(mols), size=2, replace=True, p=weights/weights.sum()))
        parent1 = mols[parent_indices[0]]
        parent2 = mols[parent_indices[1]]

        # Crossover
        child = crossover(parent1, parent2)
        if child is None:
            attempts += 1
            continue

        # Mutation
        child = mutate(child, mutation_rate=0.1)

        if child is not None:
            child_smi = Chem.MolToSmiles(child)
            if child_smi and child_smi not in [Chem.MolToSmiles(m) for m in new_mols]:
                new_mols.append(child)

        attempts += 1

    new_smiles = [Chem.MolToSmiles(m) for m in new_mols if m is not None]
    return new_smiles, [0.0] * len(new_smiles)  # Scores to be filled by proxy
