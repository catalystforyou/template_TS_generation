from rdkit import Chem
import numpy as np
from itertools import combinations
from copy import deepcopy
from scipy.spatial import Delaunay
from tqdm import trange
import os
import subprocess


def read_gjf(filename): # read gjf file and return a rdkit mol object
    atoms = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        start = False
        for line in lines:
            if start:
                parts = line.split()
                if len(parts) == 4: 
                    atom_symbol, x, y, z = parts
                    atoms.append((atom_symbol, float(x), float(y), float(z)))
            if line.strip() == "0 2" or line.strip() == '0 1':  # starting sign of the coordinates
                start = True
    mol = Chem.RWMol()
    conf = Chem.Conformer()
    for i, (atom_symbol, x, y, z) in enumerate(atoms):
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        conf.SetAtomPosition(idx, (x, y, z))
    mol.AddConformer(conf)
    return mol


def is_planar(points, tol=0.05): # check if the points are in the same plane
    if len(points) < 4:
        return True
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    u, s, vh = np.linalg.svd(points)
    return s[-1] < tol

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def rotation_matrix_from_vectors(vec1, vec2): # get the rotation matrix from vec1 to vec2
    a, b = normalize(vec1), normalize(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rodrigues_rotation_matrix(axis, theta): # get the rotation matrix around the axis with theta
    axis = normalize(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def scalar_triple_product(v1, v2, v3):
    return np.dot(v1, np.cross(v2, v3))

def is_inside_tetrahedron(p, a, b, c, d): # check if the point p is inside the tetrahedron
    vertices = np.array([a, b, c, d])
    tri = Delaunay(vertices)
    return tri.find_simplex(p) >= 0

def get_distance(mol1, mol2): # get the distance between two molecules
    min_dist = float('inf')
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    for i in range(mol1.GetNumAtoms()):
        for j in range(mol2.GetNumAtoms()):
            pos1 = conf1.GetAtomPosition(i)
            pos2 = conf2.GetAtomPosition(j)
            dist = np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def translate(mol, vector):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + vector[0], pos.y + vector[1], pos.z + vector[2]))

def rotate(mol, rotation_matrix):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        new_pos = rotation_matrix.dot(np.array([pos.x, pos.y, pos.z]))
        conf.SetAtomPosition(i, (new_pos[0], new_pos[1], new_pos[2]))

def get_carbon_positions(mol): # get the positions of all carbon atoms in the molecule
    carbon_positions = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            idx = atom.GetIdx()
            pos = mol.GetConformer().GetAtomPosition(idx)
            carbon_positions[idx] = np.array([pos.x, pos.y, pos.z])
    return carbon_positions


def test_in_tetrahedron(sh, radical, h_index, s_index, free_radical_idx, radical_neighbors, normal): # test if the radical is inside the tetrahedron
    sh_vector = (-sh.GetConformer().GetAtomPosition(h_index).x + sh.GetConformer().GetAtomPosition(s_index).x, -sh.GetConformer().GetAtomPosition(h_index).y + sh.GetConformer().GetAtomPosition(s_index).y, -sh.GetConformer().GetAtomPosition(h_index).z + sh.GetConformer().GetAtomPosition(s_index).z)
    sh_vector_length = np.linalg.norm(sh_vector)
    h_ratio = 1.45 / sh_vector_length
    sh.GetConformer().SetAtomPosition(h_index, (-h_ratio * sh_vector[0], -h_ratio * sh_vector[1], -h_ratio * sh_vector[2]))
    ratio = 3.06 / sh_vector_length
    r_target_position = (-ratio*sh_vector[0], -ratio*sh_vector[1], -ratio*sh_vector[2])
        
    
    rotation_matrix = rotation_matrix_from_vectors(normal, sh_vector) # get the rotation matrix from the normal to the sh_vector

    rotate(radical, rotation_matrix)

    test_carbon_positions = {}
    for atom in radical.GetAtoms():
        if atom.GetSymbol() == 'C':
            idx = atom.GetIdx()
            pos = radical.GetConformer().GetAtomPosition(idx)
            test_carbon_positions[idx] = np.array([pos.x, pos.y, pos.z])

    radical_vector = (r_target_position[0] - test_carbon_positions[free_radical_idx][0], r_target_position[1] - test_carbon_positions[free_radical_idx][1], r_target_position[2] - test_carbon_positions[free_radical_idx][2])

    for i in radical.GetAtoms():
        position = radical.GetConformer().GetAtomPosition(i.GetIdx())
        radical.GetConformer().SetAtomPosition(i.GetIdx(), (position.x + radical_vector[0], position.y + radical_vector[1], position.z + radical_vector[2]))

    test_carbon_positions = get_carbon_positions(radical) # get the positions of all carbon atoms in the radical

    return is_inside_tetrahedron(test_carbon_positions[free_radical_idx], test_carbon_positions[radical_neighbors[0]], test_carbon_positions[radical_neighbors[1]], test_carbon_positions[radical_neighbors[2]], [0, 0, 0])


def gen_a_combination(sh, radical_orig, reflex=False):
    print('\nStarting the generation of a combination for transition state...')
    for atom in sh.GetAtoms(): # assert there is only one S atom
        if atom.GetSymbol() == 'S':
            s_index = atom.GetIdx()
            break
    s_position = sh.GetConformer().GetAtomPosition(s_index)

    min_distance = float('inf') # find the nearest H atom to the S atom
    for atom in sh.GetAtoms():
        if atom.GetSymbol() == 'H':
            position = sh.GetConformer().GetAtomPosition(atom.GetIdx())
            distance = s_position.Distance(position)
            if distance < min_distance:
                min_distance = distance
                h_index = atom.GetIdx()

    min_distance = float('inf') # find the nearest C atom to the S atom
    for atom in sh.GetAtoms():
        if atom.GetSymbol() == 'C':
            position = sh.GetConformer().GetAtomPosition(atom.GetIdx())
            distance = s_position.Distance(position)
            if distance < min_distance:
                min_distance = distance
                c_index = atom.GetIdx()
    print('The S atom idx is:', s_index)
    print('The nearest C atom idx to the S atom is:', c_index)
    print('The nearest H atom idx to the S atom is:', h_index)


    

    s_target_position = (0, 0, 0)

    conf = sh.GetConformer()

    sh_translation = (s_target_position[0] - s_position.x, s_target_position[1] - s_position.y, s_target_position[2] - s_position.z)
    
    translate(sh, sh_translation) # place the S atom to the origin
    
    cs_vector = (-conf.GetAtomPosition(c_index).x + conf.GetAtomPosition(s_index).x, -conf.GetAtomPosition(c_index).y + conf.GetAtomPosition(s_index).y, -conf.GetAtomPosition(c_index).z + conf.GetAtomPosition(s_index).z)
    max_distance = -float('inf')
    best_i, best_j = -1, -1
    
    

    pos_s = np.array(conf.GetAtomPosition(s_index))
    pos_h = np.array(conf.GetAtomPosition(h_index))
    pos_c = np.array(conf.GetAtomPosition(c_index))

    vec_sh = pos_h - pos_s
    vec_sc = pos_c - pos_s

    normal_shc = np.cross(vec_sh, vec_sc)
    normal_shc /= np.linalg.norm(normal_shc)
    I = np.eye(3)
    reflection_matrix = I - 2 * np.outer(normal_shc, normal_shc)
    # if reflex is True, reflect the radical (create R configuration from S configuration or vice versa)
    if reflex:
        print('Reflexing the molecule to create another configuration')
        rotate(radical_orig, reflection_matrix) # the rotate function can also be used to reflect the molecule
    
    carbon_positions = get_carbon_positions(radical_orig)

    # finding the neighbors of each carbon atom
    distance_threshold = 1.7  # Å
    neighbors = {idx: [] for idx in carbon_positions}
    for idx, pos in carbon_positions.items():
        for other_idx, other_pos in carbon_positions.items():
            if idx != other_idx and np.linalg.norm(pos - other_pos) <= distance_threshold:
                neighbors[idx].append(other_idx)

    # finding the 6-membered rings (and excluding the planar ones)
    candidate_rings = []
    for combo in combinations(carbon_positions.keys(), 6):
        subgraph = {idx: [n for n in neighbors[idx] if n in combo] for idx in combo}
        if all(len(n) == 2 for n in subgraph.values()):  
            points = [carbon_positions[idx] for idx in combo]
            if is_planar(points):
                candidate_rings.append(combo)

    # locating the free radical carbon
    free_radical_candidates = []
    for idx, nlist in neighbors.items():
        if len(nlist) == 3 and not any(idx in ring for ring in candidate_rings):
            free_radical_candidates.append(idx)

    if len(free_radical_candidates) != 1:
        raise ValueError("Could not determine free radical carbon")
    else:
        free_radical_idx = free_radical_candidates[0]

    radical_neighbors = neighbors[free_radical_idx]

    v1 = np.array([carbon_positions[free_radical_idx][0] - carbon_positions[radical_neighbors[0]][0], carbon_positions[free_radical_idx][1] - carbon_positions[radical_neighbors[0]][1], carbon_positions[free_radical_idx][2] - carbon_positions[radical_neighbors[0]][2]])
    v2 = np.array([carbon_positions[free_radical_idx][0] - carbon_positions[radical_neighbors[1]][0], carbon_positions[free_radical_idx][1] - carbon_positions[radical_neighbors[1]][1], carbon_positions[free_radical_idx][2] - carbon_positions[radical_neighbors[1]][2]])
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal) # get the normal vector of the plane formed by the free radical carbon and its neighbors

    radical = deepcopy(radical_orig)
    sh_rotated = deepcopy(sh)
    if not test_in_tetrahedron(sh_rotated, radical, h_index, s_index, free_radical_idx, radical_neighbors, normal):
        normal = -normal # if the radical is not inside the tetrahedron, reverse the normal vector

    print('Performing the search...')
    for j in trange(72): # rotate the S-H bond
        radical = deepcopy(radical_orig)
        phi = j * np.pi / 36
        rotation_matrix = rodrigues_rotation_matrix(cs_vector, phi)
        sh_rotated = deepcopy(sh)
        for idx in range(sh_rotated.GetNumAtoms()):
            if idx !=  h_index:
                continue
            position = sh_rotated.GetConformer().GetAtomPosition(idx)
            new_pos = rotation_matrix.dot(np.array([position.x, position.y, position.z]))
            sh_rotated.GetConformer().SetAtomPosition(idx, (new_pos[0], new_pos[1], new_pos[2]))
        sh_vector = (-sh_rotated.GetConformer().GetAtomPosition(h_index).x + sh_rotated.GetConformer().GetAtomPosition(s_index).x, -sh_rotated.GetConformer().GetAtomPosition(h_index).y + sh_rotated.GetConformer().GetAtomPosition(s_index).y, -sh_rotated.GetConformer().GetAtomPosition(h_index).z + sh_rotated.GetConformer().GetAtomPosition(s_index).z)
        sh_vector_length = np.linalg.norm(sh_vector)
        h_ratio = 1.45 / sh_vector_length
        sh_rotated.GetConformer().SetAtomPosition(h_index, (-h_ratio * sh_vector[0], -h_ratio * sh_vector[1], -h_ratio * sh_vector[2]))
        ratio = 3.06 / sh_vector_length
        r_target_position = (-ratio*sh_vector[0], -ratio*sh_vector[1], -ratio*sh_vector[2])
            
        
        rotation_matrix = rotation_matrix_from_vectors(normal, sh_vector)

        rotate(radical, rotation_matrix)

        carbon_positions = get_carbon_positions(radical)

        radical_vector = (r_target_position[0] - carbon_positions[free_radical_idx][0], r_target_position[1] - carbon_positions[free_radical_idx][1], r_target_position[2] - carbon_positions[free_radical_idx][2])
        
        translate(radical, radical_vector)

        carbon_positions = get_carbon_positions(radical)

        pre_trans_vector = radical.GetConformer().GetAtomPosition(free_radical_idx)

        for i in range(72): # rotate the radical
            theta = i * np.pi / 36
            rotation_matrix = rodrigues_rotation_matrix(sh_vector, theta)
            radical_rotated = deepcopy(radical)
            for atom in radical_rotated.GetAtoms():
                idx = atom.GetIdx()
                pos = radical_rotated.GetConformer().GetAtomPosition(idx)
                trans_pos = (pos.x - pre_trans_vector.x, pos.y - pre_trans_vector.y, pos.z - pre_trans_vector.z)
                new_pos = rotation_matrix.dot(np.array([trans_pos[0], trans_pos[1], trans_pos[2]]))
                new_pos = (new_pos[0] + pre_trans_vector.x, new_pos[1] + pre_trans_vector.y, new_pos[2] + pre_trans_vector.z)
                radical_rotated.GetConformer().SetAtomPosition(idx, (new_pos[0], new_pos[1], new_pos[2]))
            
            dist = get_distance(sh_rotated, radical_rotated)
            if dist > max_distance:
                max_distance = dist
                best_i, best_j = i, j

                combined = Chem.CombineMols(radical_rotated, sh_rotated)
    print('The best configuration is found by rotating the radical by', best_i * 5, 'degree, and the S-H bond by', best_j * 5, 'degree')
    print('The distance between the radical and the SH molecule is', max_distance)
    print(f'The molecule is saved as combined_{int(reflex)}.mol')
    Chem.MolToMolFile(combined, f"combined_{int(reflex)}.mol")

    with open('fixed.inp', 'w') as f:
        f.write('$fix\n')
        f.write(f'  atoms:{free_radical_idx+1},{s_index+len(radical_rotated.GetAtoms())+1},{h_index+len(radical_rotated.GetAtoms())+1}\n')
        f.write('$end\n')
    
def mol_to_gaussian_input(mol_file, gaussian_input_file, functional="B3LYP-D3(BJ)", basis_set="6-31G(d)", multiplicity=2):
    # 读取MOL文件
    mol = Chem.MolFromMolFile(mol_file)
    if mol is None:
        raise ValueError("无法读取MOL文件")

    # 获取分子名称
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Molecule"

    # 获取原子符号和坐标
    atoms = mol.GetAtoms()
    conf = mol.GetConformer()
    atom_coords = [(atom.GetSymbol(), conf.GetAtomPosition(atom.GetIdx())) for atom in atoms]

    # 写入Gaussian输入文件
    with open(gaussian_input_file, 'w') as f:
        f.write(f"%NProcShared=32\n")
        f.write(f"%Mem=112GB\n")
        f.write(f"# opt=(calcfc,ts,noeigentest) freq b3lyp/6-31G(d) 5d em=gd3bj\n\n")
        f.write(f"{mol_name}\n\n")
        f.write(f"0 {multiplicity}\n")  # 假设总电荷为0，多重度为2
        for symbol, coord in atom_coords:
            f.write(f"{symbol}    {coord.x:.6f}    {coord.y:.6f}    {coord.z:.6f}\n")
        f.write("\n")

def run_gaussian(gaussian_input_file, gaussian_output_file):
    # 调用Gaussian程序并等待其完成
    result = subprocess.run(["g16", gaussian_input_file], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Gaussian Failed: {result.stderr}")

    # 将Gaussian输出重定向到指定的输出文件
    with open(gaussian_output_file, 'w') as f:
        f.write(result.stdout)

def gaussian_to_orca_input(gaussian_output_file, orca_input_file):
    # 解析Gaussian输出文件，提取优化后的几何结构
    with open(gaussian_output_file, 'r') as f:
        lines = f.readlines()

    # 查找优化后的几何结构
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if "Standard orientation:" in line:
            start_idx = i + 5
        elif start_idx and "---------------------------------------------------------------------" in line and i > start_idx:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError("无法在Gaussian输出文件中找到优化后的几何结构")

    # 提取优化后的几何结构
    atomic_number_to_symbol_map = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt",
    110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
    }   
    geometry = []
    for line in lines[start_idx:end_idx]:
        parts = line.split()
        atom_symbol = parts[1]
        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
        geometry.append((atomic_number_to_symbol_map[int(atom_symbol)], x, y, z))
    print(geometry, start_idx, end_idx)

    # 写入ORCA输入文件
    with open(orca_input_file, 'w') as f:
        f.write("! m062x RIJCOSX def2-TZVPP autoaux verytightSCF noautostart miniprint nopop\n")
        f.write("%maxcore  3500\n")
        f.write("%pal nprocs  32 end\n")
        f.write("%\cpcm\n")
        f.write("smd true\n")
        f.write('SMDsolvent "MeCN"\n')
        f.write("end\n")
        f.write("* xyz 0 2\n")
        for symbol, x, y, z in geometry:
            f.write(f"{symbol}    {x:.6f}    {y:.6f}    {z:.6f}\n")
        f.write("*\n")


def run_orca(orca_input_file, orca_output_file):
    # 调用ORCA程序并等待其完成
    result = subprocess.run(["orca", orca_input_file, ">", orca_output_file], shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"ORCA Failed: {result.stderr}")

def extract_single_point_energy(orca_output_file):
    # 从ORCA输出文件中提取单点能
    with open(orca_output_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if "FINAL SINGLE POINT ENERGY" in line:
            energy = float(line.split()[-1])
            return energy

    raise ValueError("Cannot extract single point energy from ORCA output file")

if __name__ == '__main__':

    # ----------------- Stage 1 TS generation -----------------
    radical_orig = read_gjf("Radical.gjf")
    sh = read_gjf("SH.gjf")
    gen_a_combination(sh, radical_orig, False)
    gen_a_combination(sh, radical_orig, True)
    # xtb combined_0.mol --gbsa toluene --opt --input fixed.inp --uhf 1
    xtb_result = subprocess.run(['xtb', 'combined_0.mol', '--gbsa', 'toluene', '--opt', '--input', 'fixed.inp', '--uhf', '1'], capture_output=True)
    if xtb_result.returncode != 0:
        print('Calculation failed')
    else: # `mv xtbopt.mol pre_TS_0.mol`
        print('Calculation succeeded')
        subprocess.run(['mv', 'xtbopt.mol', 'pre_TS_0.mol'])
    # xtb combined_1.mol --gbsa toluene --opt --input fixed.inp --uhf 1
    xtb_result = subprocess.run(['xtb', 'combined_1.mol', '--gbsa', 'toluene', '--opt', '--input', 'fixed.inp', '--uhf', '1'], capture_output=True)
    if xtb_result.returncode != 0:
        print('Calculation failed')
    else: # `mv xtbopt.mol pre_TS_1.mol`
        print('Calculation succeeded')
        subprocess.run(['mv', 'xtbopt.mol', 'pre_TS_1.mol'])

    # ----------------- Stage 2 Converting as Guassian input files -----------------
    mol_to_gaussian_input("pre_TS_0.mol", "TS_0.gjf")
    mol_to_gaussian_input("pre_TS_1.mol", "TS_1.gjf")
    print('The Gaussian input files are saved as TS_0.gjf and TS_1.gjf')

    # ----------------- Stage 3 Running Gaussian and ORCA -----------------
    run_gaussian("TS_0.gjf", "TS_0.log")
    run_gaussian("TS_1.gjf", "TS_1.log")
    gaussian_to_orca_input("TS_0.log", "TS_0.inp")
    gaussian_to_orca_input("TS_1.log", "TS_1.inp")
    run_orca("TS_0-orca.inp", "TS_0-orca.out")
    run_orca("TS_1-orca.inp", "TS_1-orca.out")

    # ----------------- Stage 4 Extracting single point energy -----------------
    energy_0 = extract_single_point_energy("TS_0-orca.out")
    energy_1 = extract_single_point_energy("TS_1-orca.out")
    print('The single point energy of the first configuration is', energy_0)
    print('The single point energy of the second configuration is', energy_1)
