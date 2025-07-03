'''
Plectoneme prediction code ported from the original IGOR code belonging to S.H. Kim et al. eLife, https://doi.org/10.7554/eLife.36557.
Last tested in Python 3.12.9

Author of the origianl code: Elio Abbondanzieri
Author of the current Python port: B.T. Analikwu
For contact, reach out to C.Dekker@tudelft.nl or B.T.Analikwu@tudelft.nl
Written in the Cees Dekker Lab, Department of Bionanoscience, Delft University of Technology, the Netherlands

License: CC BY-NC-SA 4.0
'''

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

## Parameters dictionary that holds experimental values of dinucleotide interactions (twist, wedge, dir))
params_dict = {'AA': {'twist': np.float64(0.6126),
        'wedge': np.float64(0.0228),      
        'direction': np.float64(1.0041),  
        'cov_twist': np.float64(0.685838),
        'cov_roll': np.float64(1.135282)},
 'AC': {'twist': np.float64(0.5533),      
        'wedge': np.float64(0.0298),      
        'direction': np.float64(-0.3588), 
        'cov_twist': np.float64(0.649127),
        'cov_roll': np.float64(0.999218)},
 'AG': {'twist': np.float64(0.5672),      
        'wedge': np.float64(0.0741),      
        'direction': np.float64(0.2621),  
        'cov_twist': np.float64(0.719114),
        'cov_roll': np.float64(1.175291)},
 'AT': {'twist': np.float64(0.5201),
        'wedge': np.float64(0.0175),
        'direction': np.float64(0.0),
        'cov_twist': np.float64(0.660374),
        'cov_roll': np.float64(0.98085)},
 'CA': {'twist': np.float64(0.6109),
        'wedge': np.float64(0.089),
        'direction': np.float64(-0.0196),
        'cov_twist': np.float64(0.97015),
        'cov_roll': np.float64(1.449952)},
 'CC': {'twist': np.float64(0.5742),
        'wedge': np.float64(0.0881),
        'direction': np.float64(0.1391),
        'cov_twist': np.float64(0.644423),
        'cov_roll': np.float64(1.107017)},
 'CG': {'twist': np.float64(0.6004),
        'wedge': np.float64(0.096),
        'direction': np.float64(0.0),
        'cov_twist': np.float64(0.95968),
        'cov_roll': np.float64(1.743733)},
 'CT': {'twist': np.float64(0.5672),
        'wedge': np.float64(0.0741),
        'direction': np.float64(-0.2621),
        'cov_twist': np.float64(0.719114),
        'cov_roll': np.float64(1.175291)},
 'GA': {'twist': np.float64(0.6196),
        'wedge': np.float64(0.0422),
        'direction': np.float64(0.6683),
        'cov_twist': np.float64(0.68059),
        'cov_roll': np.float64(1.264413)},
 'GC': {'twist': np.float64(0.5882),
        'wedge': np.float64(0.0209),
        'direction': np.float64(0.0),
        'cov_twist': np.float64(0.673715),
        'cov_roll': np.float64(0.97015)},
 'GG': {'twist': np.float64(0.5742),
        'wedge': np.float64(0.0881),
        'direction': np.float64(-0.1391),
        'cov_twist': np.float64(0.644423),
        'cov_roll': np.float64(1.107017)},
 'GT': {'twist': np.float64(0.5533),
        'wedge': np.float64(0.0298),
        'direction': np.float64(0.3588),
        'cov_twist': np.float64(0.649127),
        'cov_roll': np.float64(0.999218)},
 'TA': {'twist': np.float64(0.6528),
        'wedge': np.float64(0.0436),
        'direction': np.float64(0.0),
        'cov_twist': np.float64(1.088944),
        'cov_roll': np.float64(1.9617)},
 'TC': {'twist': np.float64(0.6196),
        'wedge': np.float64(0.0422),
        'direction': np.float64(-0.6683),
        'cov_twist': np.float64(0.68059),
        'cov_roll': np.float64(1.264413)},
 'TG': {'twist': np.float64(0.6109),
        'wedge': np.float64(0.089),
        'direction': np.float64(0.0196),
        'cov_twist': np.float64(0.97015),
        'cov_roll': np.float64(1.449952)},
 'TT': {'twist': np.float64(0.6126),
        'wedge': np.float64(0.0228),
        'direction': np.float64(-1.0041),
        'cov_twist': np.float64(0.685838),
        'cov_roll': np.float64(1.135282)}}
alphabet = ['A', 'C', 'G', 'T']
dinucleotide_alphabet = [a + b for a in alphabet for b in alphabet]  # 16 dinucleotides


def nuc2num(nucleotides):
    """
    Converts a list of nucleotides into a list of corresponding numbers in the
    range [0, 3].

    Parameters
    ----------
    nucleotides : list
        List of nucleotides to convert.

    Returns
    -------
    list
        List of numbers corresponding to the input nucleotides, where 'A' -> 0,
        'C' -> 1, 'G' -> 2, and 'T' -> 3.
    """
    return [alphabet.index(n) for n in nucleotides]


def cumMatMul3D(A, leftIsOld: bool = True):
    """
    Compute the cumulative matrix product of a 3D array.

    Parameters
    ----------
    A : ndarray
        3D array of shape (N, M, M) to compute the cumulative matrix product of.
    leftIsOld : bool, optional
        If True, compute the cumulative product as a left multiplication, i.e.,
        the product of the current matrix with the cumulative product of the
        previous matrices. If False, compute the cumulative product as a right
        multiplication, i.e., the product of the cumulative product of the
        previous matrices with the current matrix. Default is True.

    Returns
    -------
    ndarray
        3D array of shape (N, M, M) where the i-th matrix is the cumulative
        product of the first i matrices of A.
    """

    ret = np.ndarray([A.shape[0], *np.matmul(A[0, :, :], A[0, :, :]).shape])
    ret[0, :, :] = A[0, :, :]
    for i in range(1, A.shape[0]):
        if leftIsOld:
            ret[i, :, :] = np.matmul(ret[i-1, :, :], A[i, :, :])
        else:
            ret[i, :, :] = np.matmul(A[i, :, :], ret[i-1, :, :])
    return ret


def calc_DNA_path(seq: str, params_dict=params_dict):
    """
    Compute the DNA path, mean genome path, and covariance of a given sequence.

    Parameters
    ----------
    seq : str
        The DNA sequence to compute the path of.
    params_dict : dict, optional
        A dictionary mapping dinucleotide strings to dictionaries of parameters
        (twist, wedge, direction, cov_twist, cov_roll). Default is the global
        params_dict.
    Returns
    -------
    DNApath : ndarray
        2D array of shape (n_bp, 4) where the i-th row represents the position of
        the i-th basepair in 3D space.
    DNApathMG : ndarray
        2D array of shape (n_bp, 4) where the i-th row represents the mean genome
        position of the i-th basepair in 3D space.
    bpCov : ndarray
        3D array of shape (n_bp, 2, 2) where the i-th matrix is the covariance of
        the i-th basepair in 3D space.
    """
    seq_nums = np.array(nuc2num(seq))
    dinucleotides = 4*seq_nums[:-1] + seq_nums[1:]
    
    twist = np.array([params_dict[dn]["twist"] for dn in dinucleotide_alphabet])
    twists = np.array([twist[dn] for dn in dinucleotides])
    phase = np.cumsum(twists)

    wedge = np.array([params_dict[dn]["wedge"] for dn in dinucleotide_alphabet])
    direction = np.array([params_dict[dn]["direction"] for dn in dinucleotide_alphabet])
    cov_twist = np.array([params_dict[dn]["cov_twist"] for dn in dinucleotide_alphabet])
    cov_roll = np.array([params_dict[dn]["cov_roll"] for dn in dinucleotide_alphabet])

    rise = 0.339

    T = np.identity(4)
    T[2, 3] = -rise/2

    bendRot = np.ndarray((len(phase), 2, 2))  # Bending rotation matrix based on the phase
    bendRot[:, 0, 0] = np.cos(phase)
    bendRot[:, 0, 1] = -np.sin(phase)
    bendRot[:, 1, 0] = np.sin(phase)
    bendRot[:, 1, 1] = np.cos(phase)

    cov = np.zeros((len(seq)-1, 2, 2))
    cov[:, 0, 0] = cov_roll[dinucleotides]
    cov[:, 1, 1] = cov_twist[dinucleotides]

    bpCov = np.matmul(np.matmul(bendRot, cov), np.transpose(bendRot, axes=(0, 2, 1)))  # basepair covariance
    omdiv_2 = twists/2
    Romega = np.zeros((len(omdiv_2), 4, 4))
    Romega[:, 0, 0] = np.cos(omdiv_2)
    Romega[:, 0, 1] = np.sin(omdiv_2)
    Romega[:, 1, 0] = -np.sin(omdiv_2)
    Romega[:, 1, 1] = np.cos(omdiv_2)
    Romega[:, 2, 2] = 1
    Romega[:, 3, 3] = 1

    alpha = np.array([wedge[dn] for dn in dinucleotides])
    beta = np.array([direction[dn] for dn in dinucleotides]) - pi/2

    Rzplus = np.zeros((len(beta), 4, 4))
    Rzplus[:, 0, 0] = np.cos(beta)
    Rzplus[:, 0, 1] = np.sin(beta)
    Rzplus[:, 1, 0] = -np.sin(beta)
    Rzplus[:, 1, 1] = np.cos(beta)
    Rzplus[:, 2, 2] = 1
    Rzplus[:, 3, 3] = 1

    Rx = np.zeros((len(alpha), 4, 4))
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = np.cos(-alpha)
    Rx[:, 1, 2] = np.sin(-alpha)
    Rx[:, 2, 1] = -np.sin(-alpha)
    Rx[:, 2, 2] = np.cos(-alpha)
    Rx[:, 3, 3] = 1

    Rzminus = np.zeros((len(alpha), 4, 4))
    Rzminus[:, 0, 0] = np.cos(-beta)
    Rzminus[:, 0, 1] = np.sin(-beta)
    Rzminus[:, 1, 0] = -np.sin(-beta)
    Rzminus[:, 1, 1] = np.cos(-beta)
    Rzminus[:, 2, 2] = 1
    Rzminus[:, 3, 3] = 1

    startPos = np.array([0, 0, 0, 1])
    startPosMG = np.array([1, 0, 0, 1])
    n_bp = len(dinucleotides)

    T = np.identity(4)
    T[2, 3] = -rise / 2
    Q = np.matmul(np.matmul(Rzplus, Rx), Rzminus)

    # Stack T to apply it to every basepair step
    T_array = np.tile(T, (n_bp, 1, 1))

    # Matrix product sequence: T * Romega * Q * Romega * T
    step_transform = np.matmul(
        T_array,
        np.matmul(
            Romega,
            np.matmul(
                Q,
                np.matmul(Romega, T_array)
            )
        )
    )

    # Inverse of each step matrix
    Minverse = np.linalg.inv(step_transform)
    Minverse = np.insert(Minverse, 0, np.identity(4), axis=0)

    # Apply cumulative transformation
    Minverse_cum = cumMatMul3D(Minverse, leftIsOld=True)

    # Compute path
    DNApath = np.einsum('ijk,k->ij', Minverse_cum, startPos)[:, :3]
    DNApathMG = np.einsum('ijk,k->ij', Minverse_cum, startPosMG)[:, :3]
    
    return DNApath, DNApathMG, bpCov


def calc_tangent_vectors(DNApath: np.ndarray, tan_length: int = 10):
    """
    Compute tangent vectors for a DNA path

    Parameters
    ----------
    DNApath : np.ndarray
        The path of DNA in 3D space as calculated by calc_DNA_path of shape (N_bp, 3)
    tan_length: int
        The number of basepairs to compute tangent vectors for
    Returns
    -------
    tan_vectors : ndarray
        The tangent vectors of shape (N_bp, 3)
    """
    half_tan = tan_length // 2
    tan_vectors = np.zeros((len(DNApath), 3))

    for i in range(half_tan, len(DNApath) - half_tan):
        delta = DNApath[i + half_tan, :3] - DNApath[i - half_tan, :3]
        tan_vectors[i] = delta / np.linalg.norm(delta)
    return tan_vectors


def calc_curvature(seq, DNApath, DNApathMG, tan_vectors, curve_window:int = 40, params_dict: dict = params_dict):
    """
    Compute the curvature of a DNA path

    Parameters
    ----------
    seq : str
        The sequence of the DNA molecule
    DNApath : ndarray
        The path of DNA in 3D space as calculated by calc_DNA_path of shape (N_bp, 3)
    DNApathMG : ndarray
        The path of DNA in 3D space as calculated by calc_DNA_path of shape (N_bp, 3), but with the
        major groove as the reference point
    tan_vectors : ndarray
        The tangent vectors of shape (N_bp, 3)
    curve_window : int
        The number of basepairs to compute curvature for
    params_dict : dict
        A dictionary of parameters for each dinucleotide

    Returns
    -------
    curve_vectors : ndarray
        The curvature vectors of shape (N_bp, 3)
    curve_mags : ndarray
        The curvature magnitudes of shape (N_bp)
    curve_phases : ndarray
        The phase angles of the curvature of shape (N_bp)
    """
    twist = np.array([params_dict[dn]["twist"] for dn in dinucleotide_alphabet])
    
    seq_nums = np.array(nuc2num(seq))
    dinucleotides = 4*seq_nums[:-1] + seq_nums[1:]
    twists = np.array([twist[dn] for dn in dinucleotides])
    phase = np.cumsum(twists)
    half_curve = curve_window // 2
    curve_vectors = np.zeros((len(DNApath), 3))
    curve_mags = np.zeros(len(DNApath))
    curve_phases = np.zeros(len(DNApath))

    norm_vectors = (DNApathMG - DNApath)[:, :3]

    for i in range(half_curve, len(DNApath) - half_curve):
        tP = tan_vectors[i + half_curve]  # Tangent vector in the positive direction
        tM = tan_vectors[i - half_curve]  # Tangent vector in the negative direction (minus)
        curve_vec = np.cross(tP, tM)  # Curvature vector
        mag = np.linalg.norm(curve_vec)  # Curvature vector magnitude
        if mag > 0:
            curve_vec /= mag
        curve_vectors[i] = curve_vec  # Store curvature vector
        curve_mags[i] = np.arcsin(mag)  # Store curvature magnitude, which is actually the bending angle

        # Compute phase angle of curvature
        t = tan_vectors[i]
        n = norm_vectors[i]
        cos_theta = np.dot(curve_vec, n)
        sin_theta = np.dot(np.cross(n, curve_vec), t)
        curve_phases[i] = (np.arctan2(sin_theta, cos_theta) + phase[i]) % (2 * pi)

    return curve_vectors, curve_mags, curve_phases

def calc_boltzmann_prob(bpCov, curve_mags, curve_phases, curve_window: int = 40, smoothing_sigma:float = 500, circ_frac: float = 0.667, bind_length: int = 1, ave_plec_length: int = 1000, energy_offset_prefactor: float = 25):
    """
    Compute Boltzmann probabilities of curvature angles for each basepair in a given sequence

    Parameters
    ----------
    bpCov : ndarray
        Covariance matrix of basepair movements, shape (N_bp, 2, 2)
    curve_mags : ndarray
        Magnitude of curvature vector at each basepair, shape (N_bp)
    curve_window : int
        Number of basepairs to compute curvature for
    smoothing_sigma : float
        Standard deviation of the Gaussian filter to apply to the Boltzmann weights for smoothing
    circ_frac : float
        Fraction of the total length of the DNA molecule that is circularized
    bind_length : int
        Number of basepairs to exclude from the Boltzmann weights at the ends of the DNA molecule
    ave_plec_length : int
        Average length of a plectoneme, used to compute the end effects factor
    energy_offset_prefactor : float
        Energy offset prefactor, used to adjust the Boltzmann weights

    Returns
    -------
    sequence_angle_exp : ndarray
        Boltzmann weights for each basepair, shape (N_bp). When normalized, can be interpreted as a probability.
    sequence_angle_exp_smth : ndarray
        Smoothed Boltzmann weights, shape (N_bp)
    sequence_angle_energy : ndarray
        Energy corresponding to each Boltzmann weight, shape (N_bp)
    """
    e_base = (circ_frac**2) * 3000 / curve_window  # curvature energy prefactor
    energy_offset = energy_offset_prefactor - curve_window*0.334*3/4.06
    
    # Preallocate memory:
    sequence_angle_exp = np.zeros(len(seq))
    sequence_angle_energy = np.zeros(len(seq))


    local_cov = uniform_filter1d(bpCov, size=curve_window+1, axis=0, mode='nearest')
    for i in range(bind_length, len(seq) - bind_length):
        covar = local_cov[i]  # shape (2,2)

        # Rotate covariance matrix to align with curvature direction (major groove frame)
        angle = curve_phases[i]
        bend_rot = np.array([[np.cos(angle),  np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)]])
        cov_rot = bend_rot @ covar @ bend_rot.T

        # Also rotate by 45 degrees
        angle_45 = np.pi / 4
        bend_rot_45 = np.array([[np.cos(angle_45),  np.sin(angle_45)],
                                [-np.sin(angle_45), np.cos(angle_45)]])
        cov_rot_45 = bend_rot_45 @ covar @ bend_rot_45.T

         # Normalize curvature
        cnorm = curve_mags[i] / (2 * np.pi * circ_frac)

        # Boltzmann weights for bending in various directions
        Z1 = np.exp(-e_base / cov_rot[0, 0] * (1 - cnorm)**2 + energy_offset)  # along curve
        Z2 = np.exp(-e_base / cov_rot[0, 0] * (1 + cnorm)**2 + energy_offset)  # against curve
        Z3 = np.exp(-e_base / cov_rot[1, 1] * (1 - cnorm**2) + energy_offset)  # perpendicular
        Z4 = np.exp(-e_base / cov_rot_45[0, 0] * (np.sqrt(cnorm**2 / 2 + 1) - cnorm / np.sqrt(2))**2 + energy_offset)
        Z5 = np.exp(-e_base / cov_rot_45[0, 0] * (np.sqrt(cnorm**2 / 2 + 1) + cnorm / np.sqrt(2))**2 + energy_offset)
        Z6 = np.exp(-e_base / cov_rot_45[1, 1] * (np.sqrt(cnorm**2 / 2 + 1) - cnorm / np.sqrt(2))**2 + energy_offset)
        Z7 = np.exp(-e_base / cov_rot_45[1, 1] * (np.sqrt(cnorm**2 / 2 + 1) + cnorm / np.sqrt(2))**2 + energy_offset)

        sequence_angle_exp[i] = Z1 + Z2 + 2 * Z3 + Z4 + Z5 + Z6 + Z7

    # End effects factor
    p = np.arange(len(seq))
    end_effects = np.maximum(0, np.minimum(1, (p - bind_length) / ave_plec_length) *
                                np.minimum(1, (len(seq) - p - bind_length) / ave_plec_length))

    # Apply and normalize
    sequence_angle_exp *= end_effects
    sequence_angle_exp_smth = gaussian_filter1d(sequence_angle_exp, sigma=smoothing_sigma)
    sequence_angle_exp_smth /= np.mean(sequence_angle_exp_smth)

    # Convert to energy
    sequence_angle_energy = -np.log(sequence_angle_exp + 1e-12)  # avoid log(0)

    return sequence_angle_exp, sequence_angle_exp_smth, sequence_angle_energy


def predictPlectonemeProb(seq: str):
    DNApath, DNApathMG, bpCov = calc_DNA_path(seq)

    tan_vectors = calc_tangent_vectors(DNApath, tan_length=10)

    c_v, curve_mag, curve_phases = calc_curvature(seq, DNApath, DNApathMG, tan_vectors)
    sequence_angle_exp, sequence_angle_exp_smth, sequence_angle_energy = calc_boltzmann_prob(bpCov, curve_mags=curve_mag, curve_phases=curve_phases, curve_window=40)
    return sequence_angle_exp, sequence_angle_exp_smth, sequence_angle_energy


if __name__ == "__main__":
    N = 50000
    # Generate a random sequence of length N
    seq = np.random.choice(alphabet, N)

    # Insert a highly curved region at the center; This is is Curve75-2 from the eLife article.
    seq[N//2:N//2+75] = list('GATGCTCACCGCATTTCCTGAAAATTCACGCTGTATCTTGAAAAATCGACGTTTTTTACGTGGTTTTCCGTCGAA')
    seq = ''.join(seq)

    # sequence_angle_exp, sequence_angle_exp_smth, sequence_angle_energy = predictPlectonemeProb(seq)
    ### The below code does the same as the function predictPlectonemeProb(seq) in the line above


    DNApath, DNApathMG, bpCov = calc_DNA_path(seq)

    tan_vectors = calc_tangent_vectors(DNApath, tan_length=10)

    c_v, curve_mag, curve_phases = calc_curvature(seq, DNApath, DNApathMG, tan_vectors)
    sequence_angle_exp, sequence_angle_exp_smth, sequence_angle_energy = calc_boltzmann_prob(bpCov, curve_mags=curve_mag, curve_phases=curve_phases, curve_window=40)
    
    ## Some nice plots for visualization:
    
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection='3d')
    print(DNApath[:1000, 0].shape, DNApath[:, 0].shape)
    ax.plot(DNApath[:, 0], DNApath[:, 1], DNApath[:, 2])
    ax.plot(DNApathMG[:, 0], DNApathMG[:, 1], DNApathMG[:, 2])
    plt.show(block=False)
    plt.figure()
    plt.plot(sequence_angle_exp_smth / sum(sequence_angle_exp_smth))
    plt.ylabel('Boltzmann probability| smoothed', color='C0')
    plt.tick_params(axis='y', labelcolor='C0')
    twinx = plt.gca().twinx()
    twinx.plot(sequence_angle_exp / sum(sequence_angle_exp), c='C1')
    twinx.set_ylabel('Boltzmann probability', color='C1')
    twinx.tick_params(axis='y', labelcolor='C1')
    plt.tight_layout()
    plt.show()

    



