
"""
Script for calculating various spectra.

"""

import numpy as np
import scipy.sparse.linalg
from collections import OrderedDict
import sys,os
from mpi4py import MPI
import pickle
import time
import argparse
#import h5py
# Local local stuff
#from impurityModel.ed import spectra
from impurityModel.ed import finite
#import finite
from impurityModel.ed.finite import c2i
from impurityModel.ed.average import k_B
from impurityModel.ed import product_state_representation as psr

def main(e_vimp, e_cimp,
         e_vval, e_vcon,
         v_vval, v_vcon,
         e_cval, e_ccon,
         v_cval, v_ccon,
         e_v4040, v_v4043,

         radial_filename,
         ls, nBaths, nValBaths,
         n0imps, dnTols, dnValBaths, dnConBaths,
         Fcc, Fvv, Fvc, Gvc,
         xi_v, xi_c, chargeTransferCorrection,
         hField, nPsiMax,
         nPrintSlaterWeights, tolPrintOccupation,
         T, energy_cut,
         delta, deltaRIXS, deltaNIXS):
    """
    First find the lowest eigenstates and then use them to calculate various spectra.

    Parameters
    ----------
   e_vimp : list
        Energy of impurity valence orbitals.
    e_cimp : list
        Energy of impurity core orbitals.
    e_vval : float
        Energy position of valence orbital valence bath states.
    e_cval : float
        Energy position of core orbital valence bath states.
    e_vcon : float
        Energy position of valence orbital conduction bath states.
    e_ccon : float
        Energy position of core orbital conduction bath states.
    v_vval : float
        Hybridization/hopping strength of valence orbital valence bath states.
    v_cval : float
        Hybridization/hopping strength of core orbital valence bath states.
    v_vcon : float
        Hybridization/hopping strength of valence orbital conduction bath states.
    v_ccon : float
        Hybridization/hopping strength of core orbital conduction bath states.
    radial_filename : str
        File name of file containing radial mesh and radial part of final
        and initial orbitals in the NIXS excitation process.
    ls : tuple
        Angular momenta of correlated orbitals.
    nBaths : tuple
        Number of bath states,
        for each angular momentum.
    nValBaths : tuple
        Number of valence bath states,
        for each angular momentum.
    n0imps : tuple
        Initial impurity occupation.
    dnTols : tuple
        Max devation from initial impurity occupation,
        for each angular momentum.
    dnValBaths : tuple
        Max number of electrons to leave valence bath orbitals,
        for each angular momentum.
    dnConBaths : tuple
        Max number of electrons to enter conduction bath orbitals,
        for each angular momentum.
    Fcc : tuple
        Slater-Condon parameters Fcc.
    Fvv : tuple
        Slater-Condon parameters Fvv.
    Fvc : tuple
        Slater-Condon parameters Fvc.
    Gvc : tuple
        Slater-Condon parameters Gvc.
    xi_v : float
        SOC value for valence-orbitals.
    xi_c : float
        SOC value for core-orbitals.
    chargeTransferCorrection : float
        Double counting parameter
    hField : tuple
        Magnetic field.
    nPsiMax : int
        Maximum number of eigenstates to consider.
    nPrintSlaterWeights : int
        Printing parameter.
    tolPrintOccupation : float
        Printing parameter.
    T : float
        Temperature (Kelvin)
    energy_cut : float
        How many k_B*T above lowest eigenenergy to consider.
    delta : float
        Smearing, half width half maximum (HWHM). Due to short core-hole lifetime.
    deltaRIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.
    deltaNIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.


    """

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0: t0 = time.time()

    # -- System information --
    nBaths = OrderedDict(zip(ls, nBaths))
    nValBaths = OrderedDict(zip(ls, nValBaths))

    # -- Basis occupation information --
    n0imps = OrderedDict(zip(ls, n0imps))
    dnTols = OrderedDict(zip(ls, dnTols))
    dnValBaths = OrderedDict(zip(ls, dnValBaths))
    dnConBaths = OrderedDict(zip(ls, dnConBaths))

    # Changing type from list to float
    # e_v4040 = float(e_v4040)

    # -- Spectra information --
    # Energy cut in eV.
    energy_cut *= k_B*T
    # XAS parameters
    # Energy-mesh
    w = np.linspace(-25, 25, 3000)
    # Each element is a XAS polarization vector.
    epsilons = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # [[0,0,1]]
    # RIXS parameters
    # Polarization vectors, of in and outgoing photon.
    epsilonsRIXSin = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # [[0,0,1]]
    epsilonsRIXSout = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # [[0,0,1]]
    wIn = np.linspace(-10, 20, 50)
    wLoss = np.linspace(-2, 12, 4000)
    # NIXS parameters
    qsNIXS = [2 * np.array([1, 1, 1]) / np.sqrt(3), 7 * np.array([1, 1, 1]) / np.sqrt(3)]
    # Angular momentum of final and initial orbitals in the NIXS excitation process.
    liNIXS,ljNIXS = 2, 2

    # -- Occupation restrictions for excited states --
    l = ls[1]
    restrictions = {}
    # Restriction on impurity orbitals
    indices = frozenset(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
    restrictions[indices] = (n0imps[l] - 1, n0imps[l] + dnTols[l] + 1)
    # Restriction on valence bath orbitals
    indices = []
    for b in range(nValBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (nValBaths[l] - dnValBaths[l], nValBaths[l])
    # Restriction on conduction bath orbitals
    indices = []
    for b in range(nValBaths[l], nBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (0, dnConBaths[l])

    # Read the radial part of correlated orbitals
    #radialMesh, RiNIXS = np.loadtxt(radial_filename).T
    #RjNIXS = np.copy(RiNIXS)

    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2*(2*ang+1)+nBath for ang, nBath in nBaths.items())
    if rank == 0: print("#spin-orbitals:", n_spin_orbitals)
    bath_state_basis = 'special_cubic'

    # Hamiltonian
    if rank == 0: print('Construct the Hamiltonian operator...')
    hOp = get_hamiltonian_operator_using_CF(ls, nBaths, nValBaths, [Fcc, Fvv, Fvc, Gvc],
                                            [xi_v, xi_c],
                                            [n0imps, chargeTransferCorrection],
                                            hField,
                                            e_vimp, e_cimp,
                                            e_vval, e_vcon,
                                            v_vval, v_vcon,
                                            e_cval, e_ccon,
                                            v_cval, v_ccon,
                                            e_v4040, v_v4043,
                                            rank,bath_state_basis=bath_state_basis)
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0: print('{:d} processes in the Hamiltonian.'.format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0: print('Create basis...')
    basis = finite.get_basis(nBaths, nValBaths, dnValBaths, dnConBaths,
                             dnTols, n0imps)
    if rank == 0: print('#basis states = {:d}'.format(len(basis)),flush=True)
    # Diagonalization of restricted active space Hamiltonian
    es, psis = finite.eigensystem(n_spin_orbitals, hOp, basis, nPsiMax)

    if rank == 0:
        print("time(ground_state) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    # Calculate static expectation values
    finite.printThermalExpValues(nBaths, es, psis)
    finite.printExpValues(nBaths, es, psis)

    # Print Slater determinants and weights
    if rank == 0:
        print('Slater determinants/product states and corresponding weights')
        weights = []
        for i, psi in enumerate(psis):
            print('Eigenstate {:d}.'.format(i))
            print('Consists of {:d} product states.'.format(len(psi)))
            ws = np.array([ abs(a)**2 for a in psi.values() ])
            s = np.array([ ps for ps in psi.keys() ])
            j = np.argsort(ws)
            ws = ws[j[-1::-1]]
            s = s[j[-1::-1]]
            weights.append(ws)
            if nPrintSlaterWeights > 0:
                print('Highest (product state) weights:')
                print(ws[:nPrintSlaterWeights])
                print('Remaining (product state) weights: ',1-np.sum(ws[:nPrintSlaterWeights]))
                print('Corresponding product states:')
                for b in s[:nPrintSlaterWeights]:
                    print(psr.bytes2tuple(b,n_spin_orbitals))
                print('')

    # Calculate density matrix
    if rank == 0:
        print('Density matrix (in cubic harmonics basis):')
        for i, psi in enumerate(psis):
            print('Eigenstate {:d}'.format(i))
            #n = finite.getDensityMatrixCubic(nBaths, psi)
            #print('#density matrix elements: {:d}'.format(len(n)))
            #for e, ne in n.items():
            #    if abs(ne) > tolPrintOccupation:
            #        if e[0] == e[1]:
            #            print('Diagonal: (i,s) =',e[0],', occupation = {:7.2f}'.format(ne))
            #        else:
            #            print('Off-diagonal: (i,si), (j,sj) =',e,', {:7.2f}'.format(ne))
            entropy,nps,ncond = linear_entropy(ls,nValBaths,nBaths,n_spin_orbitals,psi)
            print('Linear entropy, p occ, and cond occ of state {:d} (after partial trace): = {:9.7f} {:9.7f}'.format(i,np.real(entropy),nps,ncond))
            print('')

    if rank == 0:
        nv = 2*(2*ls[0]+1) # number of impurity valence states
        nc = 2*(2*ls[1]+1) # number of impurity core states

        print('Slater determinants/product states and corresponding weights (in cubic harmonics basis):')
        uOperator = finite.getvcSlaterCondonUop(ls,Fcc=Fcc, Fvv=Fvv,
                                              Fvc=Fvc, Gvc=Gvc) #1/(r_1-r_2)
        uOp = {}
        for process,value in uOperator.items():
            uOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
        # Rotate into cubic harmonics basis
        # Make the transformation matrix
        urot = np.identity(n_spin_orbitals,dtype=np.complex128)
        if (ls[0] == 2 and ls[1] == 3 and bath_state_basis == 'special_cubic'):
          u1 = np.identity(nv,dtype=np.complex128)
          #u1pre = np.array([[0,0,np.sqrt(2.0),0,0],[1j,0,0,1,0],[1j,0,0,-1,0],[0,1,0,0,1j],[0,-1,0,0,1j]]).conj().T/np.sqrt(2.0)
          u1pre=np.array([[0,0,np.sqrt(2.0),0,0],[0,1j,0,0,1],[0,-1j,0,0,1],[1j,0,0,-1,0],[1j,0,0,1,0]]).conj().T/np.sqrt(2.0)
          u1[0:nv//2,0:nv//2] = u1pre
          u1[nv//2:,nv//2:] = u1pre

          # rotation of "special cubic"
          #u1pre = np.array([[0,0,2,0,0],[1,1j,0,1j,1],[1,-1j,0,-1j,1],[1j,1,0,-1,-1j],[1j,-1,0,1,-1j]]).conj().T/2.0
          #theta = np.pi/4
          ubath = np.identity(nv,dtype=np.complex128)
          #ubathpre = np.array([[np.sqrt(2.0),0,0,0,0],[0,1.0,0,1j,0],[0,0,1.0,0,1j],[0,1j,0,1.0,0],[0,0,1j,0,1.0]])/np.sqrt(2.0)
          #ubath[0:nv//2,0:nv//2] = ubathpre
          #ubath[nv//2:,nv//2:] = ubathpre

          #u2 = np.array([
          #    [0,-np.sqrt(1.0/7),0,0,0,0,0,np.sqrt(6.0/7),0,0,0,0,0,0],
          #    [0,0,-np.sqrt(2.0/7),0,0,0,0,0,np.sqrt(5.0/7),0,0,0,0,0],
          #    [0,0,0,-np.sqrt(3.0/7),0,0,0,0,0,np.sqrt(4.0/7),0,0,0,0],
          #    [0,0,0,0,-np.sqrt(4.0/7),0,0,0,0,0,np.sqrt(3.0/7),0,0,0],
          #    [0,0,0,0,0,-np.sqrt(5.0/7),0,0,0,0,0,np.sqrt(2.0/7),0,0],
          #    [0,0,0,0,0,0,-np.sqrt(6.0/7),0,0,0,0,0,np.sqrt(1.0/7),0],
          #    [1.0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          #    [0,np.sqrt(6.0/7),0,0,0,0,0,np.sqrt(1.0/7),0,0,0,0,0,0],
          #    [0,0,np.sqrt(5.0/7),0,0,0,0,0,np.sqrt(2.0/7),0,0,0,0,0],
          #    [0,0,0,np.sqrt(4.0/7),0,0,0,0,0,np.sqrt(3.0/7),0,0,0,0],
          #    [0,0,0,0,np.sqrt(3.0/7),0,0,0,0,0,np.sqrt(4.0/7),0,0,0],
          #    [0,0,0,0,0,np.sqrt(2.0/7),0,0,0,0,0,np.sqrt(5.0/7),0,0],
          #    [0,0,0,0,0,0,np.sqrt(1.0/7),0,0,0,0,0,np.sqrt(6.0/7),0],
          #    [0,0,0,0,0,0,0,0,0,0,0,0,0,1.0]
          #    ]).conj().T
          u2 = np.identity(nc,dtype=np.complex128)
          urot[nv+nc:2*nv+nc,nv+nc:2*nv+nc] = ubath
          urot[2*nv+nc:3*nv+nc,2*nv+nc:3*nv+nc] = ubath
        else:
          u1 = finite.get_spherical_2_cubic_matrix(spinpol=True, l=ls[0])
          u2 = finite.get_spherical_2_cubic_matrix(spinpol=True, l=ls[1])

        #print("p-orbital transformation: ",u1)
        #print("d-orbital transfomration: ",u2)
        # TODO: FIXME!

        urot[0:nv,0:nv] = u1
        urot[nv:nv+nc,nv:nv+nc] = u2
        #urot[nv+nc:2*nv+nc,nv+nc:2*nv+nc] = u1
        #urot[2*nv+nc:3*nv+nc,2*nv+nc:3*nv+nc] = u1
        clist = []
        clistback =[]
        for i in range(n_spin_orbitals):
            cdict = {}
            cdictback={}
            for j in range(n_spin_orbitals):
                if (np.abs(urot[i,j]) > 0.001):
                    cdict[((j,"c"),)] = np.conj(urot[i,j])
            clist.append(cdict)
            for j in range(n_spin_orbitals):
                if (np.abs(urot[j, i]) > 0.001):
                    cdictback[((j,"c"),)] = (urot[j,i])
            clistback.append(cdictback)
            
        print("List of transformed creation operators",clist)
        weights = []
        for i, psi in enumerate(psis): #psi is a dictionary, where keys are occupation of the orbitals and the values are the coefficients for the corresponding state
            # Transform the states
            #psirot = finite.transform_basis(psi,clist)
            print('Eigenstate {:d}.'.format(i))
            print('Consists of {:d} product states.'.format(len(psi)))
            psisum ={}
            for jkey,val in psi.items():
               orbitals = psr.bytes2tuple(jkey,n_spin_orbitals)
               #print("orbitals",orbitals)
               psinew = {psr.tuple2bytes((),n_spin_orbitals):val}
               for orbital in orbitals[-1::-1]:
                   #print("transforming orbital ",orbital)
                   psinew = finite.applyOp(n_spin_orbitals,clist[orbital],psinew)
                   #print("new state ",psinew)
               finite.addToFirst(psisum,psinew)
            cs= np.array([ a for a in psisum.values() ])
            ws = np.array([ abs(a)**2 for a in psisum.values() ])
            s = np.array([ ps for ps in psisum.keys() ])
            j = np.argsort(ws)
            cs = cs[j[-1::-1]]
            ws = ws[j[-1::-1]]
            s = s[j[-1::-1]]
            weights.append(ws)
            psireduced = {}
            norm=np.sqrt(np.sum(ws[:nPrintSlaterWeights]))
            for l in range(nPrintSlaterWeights):
                psireduced[s[l]] = cs[l]/norm 
            if nPrintSlaterWeights > 0:
                entropy,nps,ncond = linear_entropy(ls,nValBaths,nBaths,n_spin_orbitals,psisum)
                print('Linear entropy, p occ, and conduction state occ of state {:d} (after partial trace): = {:9.7f} {:9.7f} {:9.7f}'.format(i,np.real(entropy),nps,ncond))
                entropy,nps,ncond = linear_entropy(ls,nValBaths,nBaths,n_spin_orbitals,psireduced)
                print('Linear entropy, p occ, and conduction state occ of trunkated state {:d} (after partial trace): = {:9.7f} {:9.7f} {:9.7f}'.format(i,np.real(entropy),nps,ncond))
                print('Highest (product state) weights (renormalised):')
                print(ws[:nPrintSlaterWeights]/(norm**2))
                print('Highest (product state) weights:')
                print(ws[:nPrintSlaterWeights])
                print('Corresponding product states: (spin up: 0-2,6-10,16-18,22-14, p_au: 0,3 p_eu: 1-2,4-5, d_eg: 6-7,11-12, d_t2g: 8-10,13-15, b_val_au: 16,19 , b_val_eu:17-18,20-21, b_con_au: 22,25, b_con_eu: 23-24,26-27 ) ')
                for b in s[:nPrintSlaterWeights]:
                    orbitals = psr.bytes2tuple(b,n_spin_orbitals)
                    psinew = {psr.tuple2bytes((),n_spin_orbitals):1.0}
                    for orbital in orbitals[-1::-1]:
                        psinew = finite.applyOp(n_spin_orbitals,clistback[orbital],psinew)
                    psiU = finite.applyOp(n_spin_orbitals,uOp,psinew)
                    Uexp= finite.inner(psinew, psiU)
                    print(orbitals, 'U = ', np.real_if_close(Uexp))
                print('')
    return
    
    # Save some information to dis
    h5f = None
    if rank == 0:
        # Most of the input parameters. Dictonaries can be stored in this file format.
        np.savez_compressed('data', ls=ls, nBaths=nBaths,
                            nValBaths=nValBaths,
                            n0imps=n0imps, dnTols=dnTols,
                            dnValBaths=dnValBaths, dnConBaths=dnConBaths,
                            Fcc=Fcc, Fvv=Fvv, Fvc=Fvc, Gvc=Gvc,
                            xi_v=xi_v, xi_c=xi_c,
                            chargeTransferCorrection=chargeTransferCorrection,
                            hField=hField,
                            e_cimp=e_cimp,
                            e_cval=e_cval,
                            e_ccon=e_ccon,
                            e_v4040=e_v4040,
                            e_v4043=e_v4043,
                            v_cval=v_cval,
                            v_ccon=v_ccon,
                            nPsiMax=nPsiMax,
                            T=T, energy_cut=energy_cut, delta=delta,
                            restrictions=restrictions,
                            epsilons=epsilons,
                            epsilonsRIXSin=epsilonsRIXSin,
                            epsilonsRIXSout=epsilonsRIXSout,
                            deltaRIXS=deltaRIXS,
                            deltaNIXS=deltaNIXS,
                            n_spin_orbitals=n_spin_orbitals,
                            hOp=hOp)
        # Save some of the arrays.
        # HDF5-format does not directly support dictonaries.
        #h5f = h5py.File('spectra.h5','w')
        #h5f.create_dataset('E',data=es)
        #h5f.create_dataset('w',data=w)
        #h5f.create_dataset('wIn',data=wIn)
        #h5f.create_dataset('wLoss',data=wLoss)
        #h5f.create_dataset('qsNIXS',data=qsNIXS)
        #h5f.create_dataset('r',data=radialMesh)
        #h5f.create_dataset('RiNIXS',data=RiNIXS)
        #h5f.create_dataset('RjNIXS',data=RjNIXS)

    if rank == 0:
        print("time(expectation values) = {:.2f} seconds \n".format(time.time()-t0))

    # Consider from now on only eigenstates with low energy
    es = tuple( e for e in es if e - es[0] < energy_cut )
    psis = tuple( psis[i] for i in range(len(es)) )
    if rank == 0: print("Consider {:d} eigenstates for the spectra \n".format(len(es)))

    spectra.simulate_spectra(es, psis, hOp, T, w, delta, epsilons,
                             wLoss, deltaNIXS, qsNIXS, liNIXS, ljNIXS, RiNIXS, RjNIXS,
                             radialMesh, wIn, deltaRIXS, epsilonsRIXSin, epsilonsRIXSout,
                             restrictions, h5f, nBaths)

    print('Script finished for rank:', rank)

def get_hamiltonian_operator_using_CF(ls, nBaths, nValBaths, slaterCondon, SOCs,
                                      DCinfo, hField,
                                      e_vimp, e_cimp,
                                      e_vval, e_vcon,
                                      v_vval, v_vcon,
                                      e_cval, e_ccon,
                                      v_cval, v_ccon,
                                      e_v4040, v_v4043,
                                      rank,
                                      bath_state_basis='spherical'):
    """
    Return the Hamiltonian, in operator form.

    Parameters
    ----------
    ls : tuple
        Angular momenta of correlated orbitals.
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    slaterCondon : list
        List of Slater-Condon parameters.
    SOCs : list
        List of SOC parameters.
    DCinfo : list
        Contains information needed for the double counting energy.
    hField : list containing two lists
        External magnetic field.
        Elements [hv_x, hv_y, hv_z][hc_x, hc_y, hc_z]
    e_vimp : list
        Energy of impurity valence orbitals.
    e_cimp : list
        Energy of impurity core orbitals.
    e_vval : float
        Energy position of valence orbital valence bath states.
    e_cval : float
        Energy position of core orbital valence bath states.
    e_vcon : float
        Energy position of valence orbital conduction bath states.
    e_ccon : float
        Energy position of core orbital conduction bath states.
    v_vval : float
        Hybridization/hopping strength of valence orbital valence bath states.
    v_cval : float
        Hybridization/hopping strength of core orbital valence bath states.
    v_vcon : float
        Hybridization/hopping strength of valence orbital conduction bath states.
    v_ccon : float
        Hybridization/hopping strength of core orbital conduction bath states.
    bath_state_basis : str
        'spherical' or 'cubic'.
        Which basis to use for the bath states.

    Returns
    -------
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.

    """
    # Divide up input parameters to more concrete variables
    Fcc, Fvv, Fvc, Gvc = slaterCondon
    xi_v, xi_c = SOCs
    n0imps, chargeTransferCorrection = DCinfo

    
    # Calculate the U operator, in spherical harmonics basis.
    uOperator = finite.getvcSlaterCondonUop(ls,Fcc=Fcc, Fvv=Fvv,
                                              Fvc=Fvc, Gvc=Gvc) #1/(r_1-r_2)
   
    # Add SOC, in spherical harmonics basis.
    SOC2pOperator = finite.getSOCop(xi_v, l=ls[0])
    SOC3dOperator = finite.getSOCop(xi_c, l=ls[1])

    # Double counting (DC) correction values.
   
   
    dc = finite.dc_MLFT(nc_i=n0imps[ls[1]], c=chargeTransferCorrection, Fcc=Fcc,
                        nv_i=n0imps[ls[0]], Fvc=Fvc, Gvc=Gvc)
    eDCOperator = {}
    for il, l in enumerate([ls[1],ls[0]]):
        for s in range(2):
            for m in range(-l, l+1):
                eDCOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = -dc[il]

    # Magnetic field
    hHfieldOperator = {}
   

    for l in ls:
        l_i = ls.index(l)
        for m in range(-l, l+1):
            hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = hField[l_i][0]* 1 / 2.
            hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hField[l_i][0] * 1 / 2.
            hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] += -hField[l_i][1] * 1 / 2. * 1j
            hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] += hField[l_i][1] * 1 / 2. * 1j
            for s in range(2):
                hHfieldOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = hField[l_i][2] * 1 / 2 if s == 1 else -hField[l_i][2] * 1 / 2
    #finite.printOp(nBaths, hHfieldOperator, 'Magnetic field operator')

    #l = 2
    #for m in range(-l, l+1):
        #hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = hField[1][0]*1/2.
        #hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hField[1][0]*1/2.
        #hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] += -hField[1][1]*1/2.*1j
        #hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] += hField[1][1]*1/2.*1j
        #for s in range(2):
            #hHfieldOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = hField[1][2]*1/2 if s==1 else -hField[1][2]*1/2
    #finite.printOp(nBaths, hHfieldOperator, 'Magnetic field operator')

    # 4040 operator
    Operator4040 = {}
    l = ls[0]
    if (l >= 2 and e_v4040 != 0):   # Note: This formula assumes d-orbitals, TODO: generalise to f-orbitals
        for m in range(-l, l + 1):
            for s in range(2):
                Operator4040[(((l, s, m ), 'c'), ((l, s, m), 'a'))] = e_v4040*(-7/12*m**4 + 31/12*m**2 - 6/5)
    #finite.printOp(nBaths,Operator4040, '4040 Operator:')

    # 4043 operator
    Operator4043 = {}
    l = ls[0]
    if (l >= 2 and v_v4043 != 0):
        for m in range(-l, l + 1):
            for s in range(2):
                if m + 3 <= l:
                    if abs(m)>= 3 or abs(m+3)>=3:
                        v = v_v4043*3/np.sqrt(2)
                    else:
                        v = v_v4043
                    Operator4043[(((l, s, m + 3), 'c'), ((l, s, m), 'a'))] = -v*1j*np.sign(m+3/2) # change sign when abs(m+3) < abs(m)
                    Operator4043[(((l, s, m), 'c'), ((l, s, m+3), 'a'))] = v*1j*np.sign(m+3/2)
                else:
                    exit
    #finite.printOp(nBaths,Operator4043, '4043 Operator:')





    # Construct non-relativistic and non-interacting Hamiltonian, from CF parameters.
    h0_operator = get_CF_hamiltonian(ls, nBaths, nValBaths,
                                     e_vimp, e_cimp,
                                     e_vval, e_vcon,
                                     v_vval, v_vcon,
                                     e_cval, e_ccon,
                                     v_cval, v_ccon,
                                     bath_state_basis)

    # Add Hamiltonian terms to one operator.
    hOperator = finite.addOps([uOperator,
                               hHfieldOperator,
                               SOC2pOperator,
                               SOC3dOperator,
                               eDCOperator,
                               h0_operator,
                               Operator4040,Operator4043])


    if (rank == 0): finite.printOp(nBaths,hOperator,"Local Hamiltonian: ")
   # if (rank == 0): finite.printOp(nBaths,Operator4040, '4040 Operator:')
   # if (rank == 0): finite.printOp(nBaths,Operator4043, '4043 Operator:')

    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process,value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
    #print("TEST", hOp)
    return hOp


def  get_CF_hamiltonian(ls, nBaths, nValBaths,
                        e_vimp, e_cimp,
                        e_vval, e_vcon,
                        v_vval, v_vcon,
                        e_cval, e_ccon,
                        v_cval, v_ccon,
                        bath_state_basis='spherical'):
    """
    Construct non-relativistic and non-interacting Hamiltonian, from CF parameters.

    Parameters
    ----------
    ls : tuple
        Angular momenta of correlated orbitals.
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    e_vimp : list
        Energy of impurity valence orbitals.
    e_cimp : list
        Energy of impurity core orbitals.
    e_vval : float
        Energy position of valence orbital valence bath states.
    e_cval : float
        Energy position of core orbital valence bath states.
    e_vcon : float
        Energy position of valence orbital conduction bath states.
    e_ccon : float
        Energy position of core orbital conduction bath states.
    v_vval : float
        Hybridization/hopping strength of valence orbital valence bath states.
    v_cval : float
        Hybridization/hopping strength of core orbital valence bath states.
    v_vcon : float
        Hybridization/hopping strength of valence orbital conduction bath states.
    v_ccon : float
        Hybridization/hopping strength of core orbital conduction bath states.
    bath_state_basis : str
        'spherical' or 'cubic'.
        Which basis to use for the bath states.

    Returns
    -------
    h0_operator : dict
        The non-relativistic non-interacting Hamiltonian in operator form.
        Hamiltonian describes 3d orbitals and bath orbitals.
        tuple : complex,
        where each tuple describes a process of two steps (annihilation and then creation).
        Each step is described by a tuple of the form:
        (spin_orb, 'c') or (spin_orb, 'a'),
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).

    """
    # Calculate impurity Hamiltonian.
    # First formulate in cubic harmonics basis and then rotate to
    # the spherical harmonics basis.

    h_imp_operator = {}
    for l in ls:
       if l == ls[0]:
          h_imp = np.zeros((2*ls[0]+1, 2*ls[0]+1))
          #e_imp_eg = e_pimp + 2 / 3 * e_pdeltaO_imp
          #e_imp_t2g = e_pimp - 1 / 3 * e_pdeltaO_imp
          #h_imp = np.zeros((2 * l + 1, 2 * l + 1))
          #np.fill_diagonal(h_imp, (e_imp_eg, e_imp_t2g, e_imp_t2g)) Felix version
          np.fill_diagonal(h_imp, e_vimp)
       else:
          h_imp = np.zeros((2*ls[1]+1, 2*ls[1]+1))
          #e_imp_eg = e_imp + 3 / 5 * e_deltaO_imp
          #e_imp_t2g = e_imp - 2 / 5 * e_deltaO_imp
          #h_imp = np.zeros((2 * l + 1, 2 * l + 1))
          np.fill_diagonal(h_imp, e_cimp)
       # Convert to spherical harmonics basis
       u = finite.get_spherical_2_cubic_matrix(spinpol=False, l=l)#spinpo tar med både upp och ner ifall satt till true, l=1-> p/l=2->d elektroner
       h_imp = np.dot(u, np.dot(h_imp, np.conj(u.T)))
       # Convert from matrix to operator form.
       # Also add spin.
       for i, mi in enumerate(range(-l, l + 1)):
           for j, mj in enumerate(range(-l, l + 1)):
               if h_imp[i,j] != 0:
                   for s in range(2):
                       h_imp_operator[(((l, s, mi), 'c'), ((l, s, mj), 'a'))] = h_imp[i,j]


    # Bath (3d) on-site energies and hoppings.
    # Calculate hopping terms between bath and impurity.
    # First formulate the terms in the cubic harmonics basis.
    # Also introduce spin.
    h_hopp_operator = {}
    e_bath_operator = {}
    for l in ls:
       if (l == 2 and bath_state_basis == 'special_cubic'):
          #u = np.array([[0,0,2,0,0],[1,1j,0,1j,1],[1,-1j,0,-1j,1],[1j,1,0,-1,-1j],[1j,-1,0,1,-1j]]).conj().T/2.0
          u=np.array([[0,0,np.sqrt(2.0),0,0],[0,1j,0,0,1],[0,-1j,0,0,1],[1j,0,0,-1,0],[1j,0,0,1,0]]).conj().T/np.sqrt(2.0)


          #print("Special cubic basis:",u)
       else:
          u = finite.get_spherical_2_cubic_matrix(spinpol=False, l=l)
       vVal = np.zeros((2 * l +1 , 2 * l + 1))
       vCon = np.zeros((2 * l + 1, 2 * l + 1))
       eBathVal = np.zeros((2 * l + 1, 2 * l + 1))
       eBathCon = np.zeros((2 * l + 1, 2 * l + 1))
       if (l == ls[0]):
          # Felix version
          #np.fill_diagonal(vVal, (v_pval_eg, v_pval_t2g, v_pval_t2g))
          #np.fill_diagonal(vCon, (v_pcon_eg, v_pcon_t2g, v_pcon_t2g))
          #np.fill_diagonal(eBathVal, (e_pval_eg, e_pval_t2g, e_pval_t2g))
          #np.fill_diagonal(eBathCon, (e_pcon_eg, e_pcon_t2g, e_pcon_t2g))
          np.fill_diagonal(vVal, v_vval)
          np.fill_diagonal(vCon, v_vcon)
          np.fill_diagonal(eBathVal, e_vval)
          np.fill_diagonal(eBathCon, e_vcon)
       else: 
          np.fill_diagonal(vVal, v_cval)
          np.fill_diagonal(vCon, v_ccon)
          np.fill_diagonal(eBathVal, e_cval)
          np.fill_diagonal(eBathCon, e_ccon)
       # For the bath states, we can rotate to any basis.
       # Which bath state basis to use is determined selected here.
       if bath_state_basis == 'cubic':
           # One example is to keep the cubic harmonics basis for the bath states.
           # This implies the following rotation matrix:
           u_bath = np.eye(np.shape(u)[0])
       else:
           # One example is to use spherical harmonics basis for the bath states.
           # This implies the following rotation matrix:
           u_bath = u
       # Rotate the bath energies and the hopping parameters
       #vVal = np.dot(u_bath, np.dot(vVal, np.conj(u.T)))
       #vCon = np.dot(u_bath, np.dot(vCon, np.conj(u.T)))
       #eBathVal = np.dot(u_bath, np.dot(eBathVal, np.conj(u_bath.T)))
       #eBathCon = np.dot(u_bath, np.dot(eBathCon, np.conj(u_bath.T)))
       vVal = np.dot(vVal, np.conj(u.T))
       vCon = np.dot(vCon, np.conj(u.T))
       # Convert from matrix to operator form.
       # Loop over spin
       for s in range(2):
           # Loop over impurity orbitals
           for i, mi in enumerate(range(-l, l + 1)):
               # Bath state index for valence bath states.
               bi_val = s * (2 * l + 1) + i
               # Bath state index for conduction bath states.
               bi_con = 2 * (2 * l + 1) + bi_val
               # Loop over impurity orbitals
               for j, mj in enumerate(range(-l, l + 1)):
                   # Bath state index for valence bath states.
                   bj_val = s * (2 * l + 1) + j
                   # Bath state index for conduction bath states.
                   bj_con = 2 * (2 * l + 1) + bj_val
                   # Hamiltonian values related to valence bath states.
                   vHopp = vVal[i,j]
                   eBath = eBathVal[i,j]
                   if vHopp != 0:
                       h_hopp_operator[(((l, bi_val), 'c'), ((l, s, mj), 'a'))] = vHopp
                       h_hopp_operator[(((l, s, mj), 'c'), ((l, bi_val), 'a'))] = vHopp.conjugate()
                   if eBath != 0:
                       e_bath_operator[(((l, bi_val), 'c'), ((l, bj_val), 'a'))] = eBath
                   # Only add the processes related to the conduction bath states if they are
                   # in the basis.
                   if nBaths[l] - nValBaths[l] == 2*(2*l+1):
                       # Hamiltonian values related to conduction bath states.
                       vHopp = vCon[i,j]
                       eBath = eBathCon[i,j]
                       if vHopp != 0:
                           h_hopp_operator[(((l, bi_con), 'c'), ((l, s, mj), 'a'))] = vHopp
                           h_hopp_operator[(((l, s, mj), 'c'), ((l, bi_con), 'a'))] = vHopp.conjugate()
                       if eBath != 0:
                           e_bath_operator[(((l, bi_con), 'c'), ((l, bj_con), 'a'))] = eBath

    # Add Hamiltonian terms to one operator.
    h0_operator = finite.addOps([h_imp_operator,
                                 h_hopp_operator,
                                 e_bath_operator])
    #finite.printOp(nBaths,h0_operator, 'Bath hamiltonian:')

    return h0_operator


def linear_entropy(ls,nValBaths,nBaths,n_spin_orbitals,psi):
    
    nv = 2 * (2 * ls[0] + 1)  # Number of impurity valence states
    nc = 2 * (2 * ls[1] + 1)  # Number of impurity core states
    nbval = nValBaths[ls[0]]  # Number of bath valence states for the impurity valence orbitals
    nbcon = nBaths[ls[0]] - nValBaths[ls[0]]  # Number of bath conduction states for the impurity valence orbitals
    # position of the conduction bath states:
    icons = nv + nc + nbval - 1  # all indices above this one corresponds to conduction bath states

    ws = np.array([ a for a in psi.values() ])
    absws = np.array([ abs(a)**2 for a in psi.values() ])
    s = np.array([ psr.bytes2tuple(ps,n_spin_orbitals) for ps in psi.keys() ])
    j = np.argsort(absws)
    ws = ws[j[-1::-1]]
    s = s[j[-1::-1]]
    #print("absws: ",absws)
    #print("Sum: ",np.sum(absws))

    # icond = nv + nc + nbval - 1 # All indices above this one corresponds to conduction bath states
    nvalence = 0 # number of impurity valence electrons
    ncond = 0 # number of conduction electrons
    VBATH=[] # places of valence electrons in psi (both impurity and bath?)
    CORE=[] # places of core electrons in psi
    for i,slate in enumerate(s):
       nvelec = 0
       ncondelec = 0
       core=[]
       vb=[]
       for c in slate:
          if nv-1<c<nv+nc:
            core.append(c-nv)
          else:
            vb.append(c)
            if c < nv+1:
             nvelec = nvelec + 1
            elif c > icons:
             ncondelec = ncondelec + 1
       nvalence = nvalence + nvelec*np.abs(ws[i])**2
       ncond = ncond + ncondelec*np.abs(ws[i])**2
       CORE.append(core)
       VBATH.append(vb)

    #print("Number of p-electrons: ",nps)

    isort = sorted(range(len(VBATH)), key=VBATH.__getitem__)
    VBATH = [VBATH[i] for i in isort]
    CORE = [CORE[i] for i in isort]
    ws=[ws[i] for i in isort]
    
    #print("D, ws: ",D,ws)

    #print("PB: ",PB)
    corelist = []
    wslist = []

    CORE_Trace=[]
    ws_Trace=[]
    for i in range(len(VBATH)-1):
        corelist.append(CORE[i])
        wslist.append(ws[i])
        if VBATH[i] != VBATH[i+1]:
           CORE_Trace.append(corelist)
           ws_Trace.append(wslist)
           corelist = []
           wslist = []
    corelist.append(CORE[-1])
    wslist.append(ws[-1])
    CORE_Trace.append(corelist)
    ws_Trace.append(wslist)

    #print("D_trace, ws_Trace: ",D_Trace,ws_Trace)

    n_electrons = len(CORE_Trace[0][0])
    n_space = nc

    rho = np.zeros((n_space**n_electrons, n_space**n_electrons),dtype=np.complex128)
    rhovec = np.zeros((n_space**n_electrons),dtype=np.complex128)
    numvec = [n_space**(n_electrons-k-1) for k in range(n_electrons)]
    for i, state in enumerate(CORE_Trace):
        rhovec[:] = 0
        for j, slater in enumerate(state):
            b = np.dot(slater, numvec)
            rhovec[b] = ws_Trace[i][j]
        rho += np.outer(rhovec,np.conjugate(rhovec))

    print("Tr(rho) = ",np.trace(rho))
    print("Tr(rho^2) = ",np.trace(rho.dot(rho)))
    entropy = np.trace(rho - rho.dot(rho))
    return entropy,nvalence,ncond

if __name__== "__main__":
    ry2ev = 13.6057
    # Parse input parameters
    parser = argparse.ArgumentParser(description='Spectroscopy simulations')
    #parser.add_argument('radial_filename', type=str,
    #                    help='Filename of radial part of correlated orbitals.')
    parser.add_argument('--e_vimp', type=float, nargs='+', default=[0,0,0,0,0],
                        help='Energy of impurity valence orbitals.')
    parser.add_argument('--e_cimp', type=float, nargs='+', default=[0,0,0,0,0,0,0],
                        help='Energy of impurity core orbitals.')
    parser.add_argument('--e_vval', type=float, nargs='+', default=[-1,-1,-0.9,-1,-0.9],
                        help='Energy position of valence orbital valence bath states.')
    parser.add_argument('--e_vcon', type=float, nargs='+', default=[0.14,0.13,0.0,0.13,0.0],
                        help='Energy position of valence orbital conduction bath states.')
    parser.add_argument('--v_vval', type=float, nargs='+', default=[0.5,0.5,0.75,0.5,0.75],
                        help=('Hybridization/hopping strength of valence orbital valence bath states.'))
    parser.add_argument('--v_vcon', type=float, nargs='+', default=[1,0.9,0.9,0.9,0.9],
                        help=('Hybridization/hopping strength of valence orbital conduction bath states.'))
    parser.add_argument('--e_cval', type=float, nargs='+', default=[0,0,0,0,0,0,0],
                        help='Energy position of core orbital valence bath states.')
    parser.add_argument('--e_ccon', type=float, nargs='+', default=[0,0,0,0,0,0,0],
                        help='Energy position of core orbital conduction bath states.')
    parser.add_argument('--v_cval', type=float, nargs='+', default=[0,0,0,0,0,0,0],
                        help=('Hybridization/hopping strength of core orbital valence bath states.'))
    parser.add_argument('--v_ccon', type=float, nargs='+', default=[0,0,0,0,0,0,0],
                        help=('Hybridization/hopping strength of core orbital conduction bath states.'))
    parser.add_argument('--ls', type=int, nargs='+', default=[2,3],
                        help='Angular momenta of correlated orbitals.')
    parser.add_argument('--nBaths', type=int, nargs='+', default=[20, 0],
                        help='Number of bath states, for each angular momentum.')
    parser.add_argument('--nValBaths', type=int, nargs='+', default=[10, 0],
                        help='Number of valence bath states, for each angular momentum.')
    parser.add_argument('--n0imps', type=int, nargs='+', default=[0, 2],
                        help='Initial impurity occupation, for each angular momentum.')
    parser.add_argument('--dnTols', type=int, nargs='+', default=[1, 0],
                        help=('Max devation from initial impurity occupation, '
                              'for each angular momentum.'))
    parser.add_argument('--dnValBaths', type=int, nargs='+', default=[1, 0],
                        help=('Max number of electrons to leave valence bath orbitals, '
                              'for each angular momentum.'))
    parser.add_argument('--dnConBaths', type=int, nargs='+', default=[1, 0],
                        help=('Max number of electrons to enter conduction bath orbitals, '
                              'for each angular momentum.'))
    parser.add_argument('--Fcc', type=float, nargs='+', default=[5.0, 0, 9.16, 0, 6.0, 0, 4.4],
                        help='Slater-Condon parameters Fcc.')
    parser.add_argument('--Fvv', type=float, nargs='+', default=[4.0, 0, 5.0, 0, 3.8],
                        help='Slater-Condon parameters Fvv.')
    parser.add_argument('--Fvc', type=float, nargs='+', default=[3.9, 0, 3.0, 0, 1.7], 
                        help='Slater-Condon parameters Fvc.')
    parser.add_argument('--Gvc', type=float, nargs='+', default=[0, 1.6, 0, 1.5, 0, 1.2],
                        help='Slater-Condon parameters Gvc.')
    parser.add_argument('--xi_v', type=float, default=0.0069*ry2ev,
                        help='SOC value for valence-orbitals.')
    parser.add_argument('--xi_c', type=float, default=0.00808*ry2ev,
                        help='SOC value for core-orbitals.')
    parser.add_argument('--chargeTransferCorrection', type=float, default=0.0,
                        help='Double counting parameter.')
    parser.add_argument('--hField', type=float, nargs='+', default=[[0, 0, 0], [0, 0, 0.0001]],
                        help='Magnetic field, first list for valence-orbitals second for core-orbitals.')
    parser.add_argument('--nPsiMax', type=int, default=3,
                        help='Maximum number of eigenstates to consider.')
    parser.add_argument('--nPrintSlaterWeights', type=int, default=20,
                        help='Printing parameter.')
    parser.add_argument('--tolPrintOccupation', type=float, default=0.5,
                        help='Printing parameter.')
    parser.add_argument('--T', type=float, default=300,
                        help='Temperature (Kelvin).')
    parser.add_argument('--energy_cut', type=float, default=10,
                        help='How many k_B*T above lowest eigenenergy to consider.')
    parser.add_argument('--delta', type=float, default=0.2,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to short core-hole lifetime.'))
    parser.add_argument('--deltaRIXS', type=float, default=0.050,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to finite lifetime of excited states.'))
    parser.add_argument('--deltaNIXS', type=float, default=0.100,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to finite lifetime of excited states.'))
    parser.add_argument('--e_v4040', type=float, default=0.037,
                        help='4040 operator for impurity valence states.')
    parser.add_argument('--v_v4043', type=float, default=-0.004*ry2ev,
                        help='4043 operator for impurity valence states.')

    args = parser.parse_args()

    # Sanity checks
    assert len(args.ls) == len(args.nBaths)
    assert len(args.ls) == len(args.nValBaths)
    for nBath, nValBath in zip(args.nBaths, args.nValBaths):
        assert nBath >= nValBath
    for ang, n0imp in zip(args.ls, args.n0imps):
        assert n0imp <= 2 * (2 * ang + 1)  # Full occupation
        assert n0imp >= 0
    assert len(args.hField) == 2
    assert len(args.ls) == 2
    #assert args.nBaths[1] == 10 or args.nBaths[1] == 20
    #assert args.nValBaths[1] == 10

    for l in args.ls:
        if l == args.ls[0]:
            assert len(args.e_vimp) == 2*l+1
            assert len(args.e_vval) == 2*l+1
            assert len(args.e_vcon) == 2*l+1
            assert len(args.v_vval) == 2*l+1
            assert len(args.v_vcon) == 2*l+1
            assert len(args.Fvv) == 2*l+1
            assert len(args.Fvc) == 2*l+1
            assert len(args.Gvc) == 2*l+2
        elif l == args.ls[1]:
            assert len(args.e_cimp) == 2*l+1
            assert len(args.e_cval) == 2*l+1
            assert len(args.e_ccon) == 2*l+1
            assert len(args.v_cval) == 2*l+1
            assert len(args.v_ccon) == 2*l+1
            assert len(args.Fcc) == 2*l+1
    #comm = MPI.COMM_WORLD
    #rank = comm.rank
    #mpisize = comm.size
    #print("My rank, size: ",rank,mpisize)


    main(e_vimp=args.e_vimp,
         e_cimp=args.e_cimp,
         e_vval=args.e_vval,
         e_vcon=args.e_vcon,
         v_vval=args.v_vval,
         v_vcon=args.v_vcon,
         e_cval=args.e_cval,
         e_ccon=args.e_ccon,
         v_cval=args.v_cval,
         v_ccon=args.v_ccon,
         e_v4040=args.e_v4040,
         v_v4043=args.v_v4043,
         radial_filename="test.dat",
         ls=tuple(args.ls), nBaths=tuple(args.nBaths),
         nValBaths=tuple(args.nValBaths), n0imps=tuple(args.n0imps),
         dnTols=tuple(args.dnTols), dnValBaths=tuple(args.dnValBaths),
         dnConBaths=tuple(args.dnConBaths),
         Fcc=tuple(args.Fcc), Fvv=tuple(args.Fvv),
         Fvc=tuple(args.Fvc), Gvc=tuple(args.Gvc),
         xi_v=args.xi_v, xi_c=args.xi_c,
         chargeTransferCorrection=args.chargeTransferCorrection,
         hField=tuple(args.hField), nPsiMax=args.nPsiMax,
         nPrintSlaterWeights=args.nPrintSlaterWeights,
         tolPrintOccupation=args.tolPrintOccupation,
         T=args.T, energy_cut=args.energy_cut,
         delta=args.delta, deltaRIXS=args.deltaRIXS, deltaNIXS=args.deltaNIXS)


hmm= '''
u1 = finite.get_spherical_2_cubic_matrix(spinpol=True, l=l)
u2 = finite.get_spherical_2_cubic_matrix(spinpol=True, l=2) 
diagonal_matrices = [u1, u2, u1, u1]
stacked_diagonal = np.vstack(diagonal_matrices)

# Determine the dimensions of the diagonal matrices
diagonal_size = A.shape[0]  # Assuming all diagonal matrices are square

# Create a block diagonal matrix with zeros in the off-diagonal blocks
giant_matrix = np.block([
    [stacked_diagonal, np.zeros((diagonal_size * 3, diagonal_size))],
    [np.zeros((diagonal_size, diagonal_size * 3)), np.zeros((diagonal_size * 3, diagonal_size * 3))]
])
print(giant_matrix)
'''
