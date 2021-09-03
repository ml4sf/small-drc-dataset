import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_NofA(log):
    f = open(log, 'r')
    for line in f:
        if 'NAtoms' in line:
            return int(line.split()[1])
            f.close()
            break


def read_data(log, NofA):
    f = open(log, 'r')
    f_lst = f.readlines()
    f.close()
    
    starting_point = f_lst.index(' Redundant internal coordinates found in file.\n')
    
    k = 1
    coord = []
    labels = []
    while len(coord) < NofA:
        line = f_lst[starting_point+k].split(',')
        coord.append([float(line[i]) for i in range(2,5)])
        labels.append(line[0])
        k += 1
    coord = np.array(coord)
    coord = np.reshape(coord, (NofA,3))
    return coord, labels
    
def get_Z(labels):
    transl = {' H':1,' C':6, ' O':8}
    Z = []
    for i in range(len(labels)):
        Z.append(transl[labels[i]])
    
    return np.array(Z)

def coulumb_mat(coord, Z):
    dim = coord.shape[0]
    C = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            if i == j:
                C[i,j] = 0.5*la.norm(coord[i,:])**2.4
            else:
                Rij = la.norm(coord[i,:] - coord[j,:])
                C[i,j] = (Z[i]*Z[j])/Rij
    C = C*1.8897259886 # Angstrom to Bohr
    return C

def plot(atoms, coord):

	fig = plt.figure()
	ax = Axes3D(fig)		
	
	colors = []
	sizes = []
	for at in atoms:
		if at == 1:
			colors.append('green')
			sizes.append(50)
		elif at== 6:
			colors.append('black')
			sizes.append(150)
		elif at == 8:
			colors.append ('blue')
			sizes.append (170)
		else:
			colors.append('red')
			sizes.append (100)
	
	txt_color = []
	txt_size = []	
	for at in atoms:
		if at == 1:			
			txt_color.append('blue')
			txt_size.append(8)
		else:
			txt_color.append('red')
			txt_size.append(10)

	xs = []
	ys = []
	zs = []		 
	for load in coord:
		xs.append(load[0])
		ys.append(load[1])
		zs.append(0)
	
	labels = []
	for i in range(1,len(atoms)+1):
	    labels.append(i)
   #labels = np.array(labels)
   	
	ax.scatter(coord[:,0], coord[:,1], coord[:,2], c= np.array(colors),s=np.array(sizes), marker = 'o')	
	ax.view_init(30, -91)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	for x_label, y_label, z_label, label in zip(coord[:,0]-0.5, coord[:,1]+0.3, coord[:,2], np.array(labels)):
		ax.text(x_label, y_label, z_label, label, color= 'red')
		
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.show()
#	plt.savefig(filename[:-4] +'png', dpi=100)	


def main():
    log = 'LB04-IV.log'
    N = get_NofA(log)
    coord, labels = read_data(log, N)
    #print(coord)
    #print(labels)
    Z = get_Z(labels)
    #print(Z)
    #print(len(labels), len(Z))
    C = coulumb_mat(coord, Z)
    np.savetxt('C.mat', C, fmt='%8.3f')
    plot(Z, coord)
    
if __name__ == '__main__': main()
    
