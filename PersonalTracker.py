# Francesco Turci, June 2015
# http://francescoturci.wordpress.com

import pylab as pl
import sys
import time

def view_slice(Array,Slice, axis=2):
    # view a slice 
    pl.figure()
    if axis==0:
        pl.imshow(Array[Slice,:,:], interpolation='none')
    elif axis==1:
        pl.imshow(Array[:,Slice,:], interpolation='none')
    elif axis==2:
        pl.imshow(Array[:,:,Slice], interpolation='none')
    pl.colorbar()


if(len(sys.argv)<3):
    print("Usage: python PersonalTracker.py filename cutting_distance")
    sys.exit(0)


filename=sys.argv[1]
diameter=float(sys.argv[2])

print "\nHello! I am going to track %s \n\n- Reading data..."%filename
start=time.time()

# read it 8 bits by 8 bits
data=pl.fromfile(filename, dtype=pl.int8)
# find the image sizes from the filename
Sizes=filename.split('_')[4:7]
# needs filenames like Series028_decon_conf_1_256_256_256_0.019917_0.019917_0.020424.hex

# The order of the arrays is reversed: z, y, x, because of the hex encoding! (it is not my fault)
SizeZ,SizeY,SizeX=int(Sizes[-1]),int(Sizes[-2]),int(Sizes[-3])

print("\nThe image has sizes:\n\nSizeX = %d\nSizeY = %d\nSizeZ = %d\n"%(SizeX,SizeY,SizeZ))
# Give it a 3d shape!
Data=pl.reshape(data, (SizeZ,SizeY,SizeX)).astype(pl.int16)
Data[Data<0]+=256

# A simple cross shaped kernel:
Cross=[[
    [0,1,0],
    [1,1,1],
    [0,1,0],],
   [[0,1,0],
    [1,1,1],
    [0,1,0],],
   [[0,1,0],
    [1,1,1],
    [0,1,0],]
     ]
# many possible filters, morphological operators and measurements to try!
from scipy.ndimage.filters import maximum_filter, gaussian_filter,rank_filter, median_filter
from scipy.ndimage.morphology import grey_opening,grey_closing, grey_erosion
from scipy.ndimage.measurements import label,center_of_mass,maximum_position


# ===================== MANIPULATING THE IMAGE ==================
# remornalising the input. 
# If needed, a nonlinear transformation can be performed changing the Exponent variable
# Larger exponennts suppress the low intensity region of the spectrum 

Exponent=1.
Data=((Data/float(Data.max()*1) )**Exponent*255).astype(int)

print "- Morphological Opening..."
Step1=grey_opening(Data, size=(3,3,3))
pl.save("Step1", Step1)

print "- Morphological Erosion..."
Step2=grey_erosion(Step1, structure=Cross)
pl.save("Step2", Step2)

# Remark: one could keep on with other transformations, other kernels and so on
# To do so, I would reccomend to use ipython, and eventually load the partial results 

FinalStep=Step2
# Uncomment to view a slice
# view_slice(Step,200)
# pl.savefig("test.png")
print "- Finding maxima..."
local_max=maximum_filter(FinalStep, size=(diameter/2, diameter/2, diameter/2))==FinalStep
Labelled, Num=label(local_max)
print "\n====> Found", Num,"maxima\n"

# if too many centres are found, the memory will explode at the next operation
if(Num>15000):
    print "Too many! I quit."
    sys.exit(0)

print "- Detecting maxima positions..."

# =================== FINDING THE CENTRES ===============
Positions=pl.array(center_of_mass(local_max,labels=Labelled, index=range(Num)))
Positions=Positions[1:]

# =================== REMOVE OVERLAPS ===================
from scipy.spatial.distance import pdist,squareform
N=Positions.shape[0]

dists=squareform(pdist(Positions))
# exclude the case of self-distance
pl.fill_diagonal(dists, pl.inf)

print "- Cutting out particles with at least two overlaps at distance <", diameter,"..."
test= (dists<diameter)
to_be_deleted=[]
for p in range(N):
    # removing only double overlaps
    if(pl.sum(test[p][p:])>1):
        to_be_deleted.append(p)
overlaps=pl.array(to_be_deleted)
No_overlaps_r1=pl.delete(Positions,overlaps, axis=0)

# Do it again: c
# omment this part if you are fine with particles that do overlap with only one other particle

dists=squareform(pdist(No_overlaps_r1))
# exclude the case of self-distance
pl.fill_diagonal(dists, pl.inf)

print "- Cutting out particles with overlaps at distance <", diameter,"..."
test= (dists<diameter)
to_be_deleted=[]
for p in range(No_overlaps_r1.shape[0]):
    if(pl.any(test[p][p:])):
        to_be_deleted.append(p)
overlaps=pl.array(to_be_deleted)
No_overlaps_r2=pl.delete(No_overlaps_r1,overlaps, axis=0)


print "\n====> Detected",No_overlaps_r2.shape[0],"particles.\n"

# ======================== SAVING THE RESULT ===================== 
# reorder the columns
z,y,x=No_overlaps_r2[:,0],No_overlaps_r2[:,1],No_overlaps_r2[:,2]

outfile="Detected_"+filename.split('_')[0][-3:]+".txt"
pl.savetxt(outfile,zip(x,y,z), fmt="%g")
print "Saved the output in ", outfile

print "Total execution time:", time.time()-start

