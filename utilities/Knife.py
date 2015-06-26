import pylab as pl
import argparse
from colorama import Fore as F
# install with :
# pip install colorama

parser=argparse.ArgumentParser(description="Trim overlapped particles, in two possible ways!",formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("filename", type=str, help="3 columns data. Comments with '#' are permitted ")
parser.add_argument("mode", type=int, help="Two options\n    1 : a la Colloids \n    2 : filter the double overlaps first" )
parser.add_argument("rcut", type=float, help="cutoff distance")
args=parser.parse_args()
filename=args.filename
mode=args.mode
cutoff=args.rcut

print "\nLoading data from file "+F.CYAN+" "+filename+F.RESET+"\n"
Data=pl.loadtxt(filename)
N=Data.shape[0]

from scipy.spatial.distance import pdist,squareform
# compute distances
dists=squareform(pdist(Data))
# exclude the case of self-distance
pl.fill_diagonal(dists, pl.inf)
test= (dists<cutoff)

if(mode==1):
    picked=[]
    for p in range(N):
        if pl.any(test[p,:]):
            test[:,p]=False
            test[p,:]=False
        else:
            picked.append(p)
        No_overlaps=Data[picked]

if(mode==2):
    print "- Cutting out particles with at least two overlaps at distance <", cutoff,"..."
    picked=[]
    for p in range(N):
        # removing only double overlaps
        if(pl.sum(test[p][p:])>1):
            pass
        else:
            picked.append(p)
    No_double_overlaps=Data[picked]
    # Do it again, now for the single overlaps
    dists=squareform(pdist(No_double_overlaps))
    # exclude the case of self-distance
    pl.fill_diagonal(dists, pl.inf)
    print "- Cutting out particles with overlaps at distance <", cutoff,"..."
    test= (dists<cutoff)
    picked_again=[]
    for p in range(No_double_overlaps.shape[0]):
        if(pl.any(test[p][p:])):
            pass
        else:
            picked_again.append(p)
    No_overlaps=No_double_overlaps[picked_again]

print "Saving file "+F.CYAN+"Trimmed_"+filename+F.RESET
pl.savetxt("Trimmed_"+filename,No_overlaps, fmt="%g")
print "\n====> Removed"+F.GREEN,N-No_overlaps.shape[0],F.RESET+"of the initial", N,"particles. "+F.RED,No_overlaps.shape[0],F.RESET+"remaining."

