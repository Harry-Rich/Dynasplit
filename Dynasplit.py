import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from tqdm import tqdm  
from MDAnalysis.transformations import wrap, unwrap


# Small class for centre of mass calculation using MDAnalysis bond information
class BeadGroup(object):
    def __init__(self, groups):
        self._groups = groups

    def __len__(self):
        return len(self._groups)

    @property
    def positions(self):
        return np.array([g.center_of_mass(unwrap = True) for g in self._groups], dtype=np.float32)

    @property
    def universe(self):
        return self._groups[0].universe
    

class Dynasplit:
    """
    Class to perform decomposition of a molecular dynamics trajectory into independent rotational and translational motion.

    NVT or NVE trajectories only 
    No triclinic cells
    """

    def __init__(self, u):
        """
        u: MDAuniverse

        """
        self.first_frame = u.trajectory[0].positions
        self.dimensions = u.trajectory[0].dimensions[0:3]
        self.u = u
        self.n_atoms = len(u.atoms)
        self.n_frames = len(u.trajectory)
        self.topology_atoms = u.atoms

    
    def decompose(self,masses = None,indices = None,slower = False, atom_types = None):
        rot_traj = np.zeros((self.n_frames,self.n_atoms,3))
        trans_traj = np.zeros((self.n_frames,self.n_atoms,3))

        rot_traj[0] = self.first_frame
        trans_traj[0] = self.first_frame

        if indices is None:
            try:
                indices = np.array([frag.indices for frag in self.u.atoms.fragments])
            except:
                "Molecules not defined as fragments, update universe or define indices"

        if masses is None:
            try:
                masses = self.u.atoms.masses
            except:
                "Molecules masses not defined in universe, update universe or define masses"

        for i,ts in tqdm(enumerate(self.u.trajectory), desc="Centering molecules"):
            if slower==True: 
                if atom_types is None:
                    raise Exception('All atom types in trajectory must be specified using MDAnalysis typing')
                Com = self.u.select_atoms(atom_types)
                com_groups = Com.fragments 
                c = BeadGroup(com_groups)
                com_pos = c.positions
            else: 
                frac_coords = (ts.positions % self.dimensions) / self.dimensions
                com_pos = pseudo_com_method(frac_coords,indices,masses,self.dimensions)

            if i != 0: 
                com_disp = com_pos - com_pos_prev_frame
                rot_traj[i] = self.u.trajectory[i].positions - np.repeat(com_disp,12,axis=0)
                trans_traj[i] = trans_traj[i-1] + np.repeat(com_disp,12,axis=0)

            com_pos_prev_frame = com_pos
                
        self.rot_traj = rot_traj
        self.trans_traj = trans_traj


    def write(self,to_write= 'both', trans_file = "translation.dcd", rot_file = "rotation.dcd"):

        full_dim = np.concatenate((self.dimensions, [90, 90, 90]))
        if to_write == 'both':
            u_rot = mda.Merge(self.topology_atoms)
            u_trans = mda.Merge(self.topology_atoms)

            u_trans.load_new(self.trans_traj)
            u_rot.load_new(self.rot_traj)

            print(f"Saving translation dcd to: {trans_file}")
            self._dcd_writer(u_trans, trans_file,full_dim)

            print(f"Saving rotation dcd to: {rot_file}")
            self._dcd_writer(u_rot, rot_file,full_dim)


        elif to_write == 'rotation':
            u_rot = mda.Merge(self.topology_atoms)
            u_rot.load_new(self.rot_traj)
            print(f"Saving rotation dcd to: {rot_file}")
            _dcd_writer(u_rot, rot_file,self.dimensions)

        elif to_write == 'translation':
            u_trans = mda.Merge(self.topology_atoms)
            u_trans.load_new(self.trans_traj)
            print(f"Saving translation dcd to: {trans_file}")
            self._dcd_writer(u_trans, trans_file,full_dim)

        else: 
            raise Exception("to_write must be 'both', 'translation' or 'rotation'")


    @staticmethod
    def _dcd_writer(u, file_name,dimensions):
        with mda.Writer(file_name, u.atoms.n_atoms) as writer:
            for ts in u.trajectory:
                ts.dimensions = dimensions
                writer.write(u.atoms)


# ---------- Helper functions. ----------------


def weighted_average(data, masses):
    """
    data: shape (n_mols, atoms_per_mol, 3)
    masses: shape(n_mols, atoms_per_mol)

    returns: 
        shape(n_mols,3)
    """

    weights_expanded = masses[:, :, np.newaxis]  


    weighted_coords = data * weights_expanded    
    sum_weighted = weighted_coords.sum(axis=1)  
    sum_weights = weights_expanded.sum(axis=1)   
    weighted_average = sum_weighted / sum_weights 

    return weighted_average


def pseudo_com_method(frac_coords,indices,masses,dimensions):
    '''
    Calculate the pseudo-centre of mass for molecular fragments in fractional coordinates.

    This function implements the pseudo-centre of mass method for molecules in periodic systems, as described in 
    DOI:10.1063/5.0260928 .

    Parameters
    ----------
    frac_coords : MDAnalysis universe trajectory array
        Fractional coordinates of all atoms in the system, assumed to be of shape (n_atoms, 3).
    indices : array-like of shape (n_molecules, atoms_per_molecule)
        List of index arrays defining each molecule in the system (1-based indices; internally converted to 0-based).
    masses : array-like
        Atomic masses corresponding to the atoms in each molecule (either single array of all 3 masses, or array shape n_molecules,atom_per_mol).
    dimensions: array-like
        trajectory dimensions u.trajectory.dimensions[0:3]


    Returns
    -------
    corrected_com : np.ndarray
        Cartesian coordinates of the pseudo-centres of mass for each molecule, corrected for periodic boundary conditions.
    '''
    n_molecules = indices.shape[0]
    atoms_per_mol = indices.shape[1]

    if (masses.shape != np.array([n_molecules,atoms_per_mol])).all():
        try: 
            masses = masses.reshape(n_molecules, atoms_per_mol)
        except:
            print("masses and indices shapes do not compute")



    s_coords = frac_coords[indices]
    theta = s_coords * (2 * np.pi)
    xi = np.cos(theta)
    zeta = np.sin(theta)

    xi_bar = weighted_average(xi,masses)
    zeta_bar = weighted_average(zeta, masses)

    theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
    new_s_coords = theta_bar / (2 * np.pi)

    # Implementation of pseudo-centre of mass approach to centre of mass calculation (see DOI:10.1063/5.0260928 ).
    pseudo_com_recentering = ((s_coords - (new_s_coords + 0.5)[:,np.newaxis, :]) % 1)
    com_pseudo_space = weighted_average(pseudo_com_recentering, masses)
    corrected_com = ((com_pseudo_space + (new_s_coords + 0.5)) % 1) * dimensions

    return corrected_com