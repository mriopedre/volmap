#%%
import MDAnalysis as md
import numpy as np
from tqdm import tqdm
from MDAnalysis.analysis import align

# %%

class Density3D():
    def __init__(self, u):
        self.u = u

    def align_trajectories(self, sel, ref_frame=0, in_memory=True, weights=None):
        """Align the trajectory based on a selection.

        Args:
            sel (str): Selection string for atoms to align.
            ref_frame (int, optional): Frame to use as reference. Defaults to 0 (first frame).
            in_memory (bool, optional): If True, keep all frames in memory. Defaults to True.
            weights (array-like, optional): Weights for the alignment. Defaults to None.
        """
        # Align trajectories based on a selection

        ref = self.u.copy()  # independent Universe with identical topology & coords
        ref.trajectory[ref_frame]  # fix the reference at the chosen frame

        # Set up the aligner. If `outfile` is given, MDAnalysis will write as it goes.
        aligner = align.AlignTraj(
            self.u,
            ref,
            select=sel,
            in_memory=in_memory,
            weights=weights,
        )

        aligner.run()

    def generate_grid(self, grid_size=0.5):
        """Generate a 3D grid for density estimation.
        Stores the grid in self.grid and metadata in self.grid_meta.

        Args:
            grid_size (float, optional): Size of the grid cells (voxels). Defaults to 0.5.
        """

        # This grid is based on the box size at frame 0.
        # i.e. it has no support for changing box sizes.
        # If your trajectory has changing box sizes, you might want to modify this.
        # Determine box size from frame 0 used to calculate occupancy

        ts = self.u.trajectory[0]  
        Lx, Ly, Lz = ts.dimensions[:3]
        min_bound = np.array([0.0, 0.0, 0.0])
        max_bound = np.array([Lx, Ly, Lz])
        
        # Create the grid
        x = np.arange(min_bound[0], max_bound[0], grid_size)
        y = np.arange(min_bound[1], max_bound[1], grid_size)
        z = np.arange(min_bound[2], max_bound[2], grid_size)

        self.grid = np.zeros((len(x), len(y), len(z)), dtype=np.float32)
        self.x, self.y, self.z = x, y, z

        self.grid_meta = {
            "origin": np.array([x[0], y[0], z[0]], dtype=float),
            "end": np.array([x[-1], y[-1], z[-1]], dtype=float),
            "spacing": float(grid_size),
            "shape": self.grid.shape,
            "axes": (x, y, z),
        }

    def compute_density(self, ligand, grid_size=0.5):
        """
        Compute the 3D density of a ligand in a box.
        3D array representing the density grid. It is stored in self.grid.
        self.grid must exist for this function to work, so generate_grid must be called first.
        Args:
            ligand (str or AtomGroup): Selection string or AtomGroup for the ligand.
            grid_size (float, optional): Size of the grid cells (voxels). it generated the grid through generate_grid. Defaults to 0.5.
        """

        # Compute the 3D density of the ligand selection around the center selection
        if isinstance(ligand, str): #make sure the ligand is an atomgroup
            ligand = self.u.select_atoms(ligand)

        nx, ny, nz = self.grid.shape

        for ts in tqdm(self.u.trajectory):
            # Get the positions of the ligand atoms
            positions = ligand.positions
            # Determine which grid cells the ligand atoms fall into
            indices = np.floor((positions - (self.x[0], self.y[0], self.z[0])) / grid_size).astype(int)

            # Increment the density in the corresponding grid cells (one for each atom within the voxel)
            # Mask to avoid out of bounds errors           
            mask = (
                (indices[:,0] >= 0) & (indices[:,0] < nx) &
                (indices[:,1] >= 0) & (indices[:,1] < ny) &
                (indices[:,2] >= 0) & (indices[:,2] < nz)
            )

            ii = indices[mask]
            np.add.at(self.grid, (ii[:,0], ii[:,1], ii[:,2]), 1.0)


    def normalize_density(self):
        """Normalize the density grid to convert counts to percentage.
        """
        total_frames = len(self.u.trajectory)
        self.grid /= total_frames
        self.grid *= 100  # convert to percentage
    
    def write_dx(self, outfile):
        """
        Write self.grid to an OpenDX (.dx) scalar field VMD can load.
        Assumes axes are orthogonal with uniform spacing.
        """

        origin = self.grid_meta["origin"]  # Å
        spacing = self.grid_meta["spacing"]
        nx, ny, nz = self.grid.shape

        with open(outfile, "w") as f:
            # grid definition
            f.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
            f.write(f"origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
            f.write(f"delta {spacing:.6f} 0.000000 0.000000\n")
            f.write(f"delta 0.000000 {spacing:.6f} 0.000000\n")
            f.write(f"delta 0.000000 0.000000 {spacing:.6f}\n")
            f.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")

            # data header
            nitems = nx * ny * nz
            f.write(f"object 3 class array type double rank 0 items {nitems} data follows\n")

            # write values in x-fastest order (ix→iy→iz), a few per line
            per_line = 6
            c = 0
            line = []
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        line.append(f"{float(self.grid[ix, iy, iz]):.6e}")
                        c += 1
                        if c % per_line == 0:
                            f.write(" ".join(line) + "\n")
                            line = []
            if line:
                f.write(" ".join(line) + "\n")

    def write_gro(self, outfile):
        """Write the first frame of the Universe to a .gro file.
        Since it is aligned, it should not matter much which frame is saved.
        """
        self.u.trajectory[0]  # go to first frame
        with md.Writer(outfile, self.u.atoms.n_atoms) as W:
            W.write(self.u.atoms)

    def calculate_and_save_density(self, center_sel, ligand_sel, grid_size=0.5, dx_out="density.dx", gro_out="aligned_frame.gro"):
        """Calculate the 3D density of a ligand selection around a center selection,
        and save the density grid to a .dx file and an aligned frame to a .gro file.
        Args:
            center_sel (str): Selection string for atoms to align.
            ligand_sel (str or AtomGroup): Selection string or AtomGroup for the ligand.
            grid_size (float, optional): Size of the grid cells (voxels). Defaults to 0.5.
            dx_out (str, optional): Output .dx file path. Defaults to "density.dx".
            gro_out (str, optional): Output .gro file path. Defaults to "aligned_frame.gro".
        """
        print(f"Aligning trajectory to {center_sel}...")
        self.align_trajectories(center_sel)
        self.generate_grid(grid_size)
        print(f"Computing density for ligands ...")
        self.compute_density(ligand_sel, grid_size)
        self.normalize_density()
        print(f"Writing density to {dx_out}...")
        self.write_dx(dx_out)
        print(f"Writing aligned frame to {gro_out}...")
        self.write_gro(gro_out)

    def calculate_and_save_density_no_realign(self, ligand_sel, grid_size=0.5, dx_out="density.dx"):
        """Calculate the 3D density of a ligand selection without aligning, for speed purposes,
        and save the density grid to a .dx file .
        Args:
            ligand_sel (str or AtomGroup): Selection string or AtomGroup for the ligand.
            grid_size (float, optional): Size of the grid cells (voxels). Defaults to 0.5.
            dx_out (str, optional): Output .dx file path. Defaults to "density.dx".
        """
        self.generate_grid(grid_size)
        print(f"Computing density for ligands ...")
        self.compute_density(ligand_sel, grid_size)
        self.normalize_density()
        print(f"Writing density to {dx_out}...")
        self.write_dx(dx_out)
       

#%%
# Example usage:
# Load your trajectory an create the object
u = md.Universe('example/topol.gro', 'example/traj.xtc')
density_estimator = Density3D(u)

# Calculate and save the density
# center_sel: selection for alignment (e.g., protein CA atoms)
# ligand_sel: selection for density calculation (e.g., all non-protein atoms)
# grid_size: size of the grid cells in Å - smaller, finer grid, more resolution, but slower
# dx_out: output .dx file for density
# gro_out: output .gro file for an aligned frame to overlay the density

density_estimator.calculate_and_save_density(
    center_sel='protein and name CA',
    ligand_sel='not protein',
    grid_size=1,
    dx_out='example/density.dx',
    gro_out='example/aligned_frame.gro'
)
