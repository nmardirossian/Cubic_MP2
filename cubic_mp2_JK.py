import pyscf
import numpy
import time
import pyfftw
import ase
import sys
import fft_cython

from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import df as pbcdf
from pyscf.pbc import tools as pbctools
from pyscf import lib as pylib
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

cell=pbcgto.Cell()

#Carbon
boxlen=6.74
ase_atom=ase.build.bulk('C', 'diamond', a=boxlen)
cell.a=ase_atom.cell
cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.basis='gth-szv'
cell.charge=0
cell.dimension=3
cell.incore_anyway=False
#cell.ke_cutoff=8.0
cell.gs=numpy.array([int(sys.argv[1]),int(sys.argv[1]),int(sys.argv[1])])
cell.max_memory=8000
cell.pseudo='gth-pade'
#cell.rcut
cell.spin=0
cell.unit='B'
cell.verbose=10
cell.build()

#mf=pbchf.KRHF(cell, kpts=cell.make_kpts([1,1,1]), exxdiv=None)
mf=pbchf.RHF(cell, exxdiv=None)
mf.conv_tol=1e-10
mf.conv_tol_grad=1e-8
mf.diis=True
mf.diis_space=20
mf.direct_scf_tol=1e-14
mf.init_guess='minao'
mf.max_cycle=200
mf.max_memory=8000
mf.verbose=10
mf.scf()

#mo=mf.mo_coeff[0]
#orben=mf.mo_energy[0]
mo=mf.mo_coeff
orben=mf.mo_energy
nao=cell.nao_nr()
#kpts=mf.kpts
kpts=numpy.array([[0.,0.,0.]])
nelec=cell.nelectron
nocc=nelec/2
nvirt=nao-nocc
orben=orben.reshape(-1,1)

tauarray=[-0.0089206000, -0.0611884000, -0.2313584000, -0.7165678000, -1.9685146000, -4.9561668000, -11.6625886000]
weightarray=[0.0243048000, 0.0915096000, 0.2796534000, 0.7618910000, 1.8956444000, 4.3955808000, 9.6441228000]
NLapPoints=len(tauarray)

with_df=pbcdf.FFTDF(cell, kpts)
coords=cell.gen_uniform_grids(with_df.gs)
gs=with_df.gs[0]
griddim=2*with_df.gs+1
dim=2*gs+1
ngs=dim*dim*dim
smalldim=int(numpy.floor(dim/2.0))+1

aoR=with_df._numint.eval_ao(cell, coords, kpts[0])[0]
moR=numpy.asarray(pylib.dot(mo.T, aoR.T), order='C')
moRocc=moR[:nocc]
moRvirt=moR[nocc:]

kptijkl=pbcdf.fft_ao2mo._format_kpts(kpts)
kpti, kptj, kptk, kptl=kptijkl
q=kptj-kpti
coulG=pbctools.get_coulG(cell, q, gs=with_df.gs)
coulGsmall=coulG.reshape([dim,dim,dim])
coulGsmall=coulGsmall[:,:,:smalldim].reshape([dim*dim*smalldim])
#coulGsmall=coulGsmall.copy(order='C')

pyfftw.interfaces.cache.enable()
def nm_fft(inp):
    return pyfftw.interfaces.numpy_fft.fftn(inp.reshape(griddim),griddim,planner_effort='FFTW_MEASURE').reshape(ngs)

def nm_ifft(inp):
    return pyfftw.interfaces.numpy_fft.ifftn(inp.reshape(griddim),griddim,planner_effort='FFTW_MEASURE').reshape(ngs)

Jtime=time.time()
EMP2J=0.0
for i in range(NLapPoints):
#for i in range(1):
    print EMP2J.real

    moRoccW=moRocc*numpy.exp(-orben[:nocc]*tauarray[i]/2.)
    moRvirtW=moRvirt*numpy.exp(orben[nocc:]*tauarray[i]/2.)

    gfunc=pylib.dot(moRoccW.T,moRoccW)
    gbarfunc=pylib.dot(moRvirtW.T,moRvirtW)
    ffunc=gfunc*gbarfunc

    gfunc=gbarfunc=None
    moRoccW=moRvirtW=None

    f2func=numpy.zeros((ngs,ngs),dtype='float64')

    print "Entering cython J"
    timeit=time.time()
    fft_cython.getJ(dim,ngs,ffunc,f2func,coulGsmall)
    print "Cython J call took: ", time.time()-timeit

    ffunc=None
    f2func=f2func*f2func.T
    Jint=numpy.sum(f2func)
    EMP2J-=2.*weightarray[i]*Jint*(cell.vol/ngs)**2.

    f2func=None

print "Took this long for J: ", time.time()-Jtime
EMP2J=EMP2J.real
print "EMP2J: ", EMP2J

Ktime=time.time()
EMP2K=0.0
for i in range(NLapPoints):
#for i in range(1):
    print EMP2K.real

    moRoccW=moRocc*numpy.exp(-orben[:nocc]*tauarray[i]/2.)
    moRvirtW=moRvirt*numpy.exp(orben[nocc:]*tauarray[i]/2.)

    gfunc=pylib.dot(moRoccW.T,moRoccW)
    gbarfunc=pylib.dot(moRvirtW.T,moRvirtW)

    moRoccW=moRvirtW=None

    f2func=numpy.zeros((ngs,ngs),dtype='float64')

    print "Entering cython K"
    timeit=time.time()
    fft_cython.getK1(dim,ngs,gfunc,gbarfunc,f2func,coulGsmall)
    print "Cython K1 call took: ", time.time()-timeit

    gfunc=gbarfunc=None

    f2funcdiag=numpy.diagonal(f2func)
    f2func=f2func+f2func.T
    numpy.fill_diagonal(f2func,f2funcdiag)
    f2func=f2func.astype(numpy.complex128)
    for j in range(ngs):
        f2func[j]=nm_fft(f2func[j]).conj()
    for j in range(ngs):
        f2func[:,j]=nm_fft(f2func[:,j])
    EMP2K-=-(cell.vol/ngs)*(cell.vol/(ngs*ngs))*weightarray[i]*numpy.sum(numpy.diagonal(f2func)*coulG)

    f2func=None

print "Took this long for K: ", time.time()-Ktime
EMP2K=EMP2K.real
print "EMP2K: ", EMP2K

EMP2=EMP2J+EMP2K
print "EMP2: ", EMP2
