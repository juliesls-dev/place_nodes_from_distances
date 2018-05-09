import random
import numpy as np
import scipy
import scipy.spatial

if __name__ == "__main__":
	nodes_real = np.array([
		[4.761842989061976, 6.403160183594165],
		[8.510013601451424, 2.61225272409742],
		[2.0519381313967955, 9.868191953942127],
		[6.165144059392952, 9.000449233772388],
		[9.982920868180624, 7.600419652524294],
		[2.091470580324002, 9.346471254630824],
	])

	s = 10000
	nodes = s * np.random.rand(6, 2)

	distances = scipy.spatial.distance_matrix(nodes_real, nodes_real)

	niter_max = 1000
	update_thr = 1e-3
	k = 1
	for iteration in range(niter_max):
		moved = 0
		for node in range(nodes.shape[0]):
			P = np.empty((5, 2))
			
			i = 0
			for neigh in range(nodes.shape[0]):
				if neigh == node:
					continue
				neigh_node = nodes[node] - nodes[neigh]
				u = neigh_node / np.linalg.norm(neigh_node)
				P[i] = nodes[neigh] + u * k * distances[node][neigh]

				i += 1

			new_node = np.mean(P, axis=0)
			moved += np.linalg.norm(new_node - nodes[node])
			nodes[node] = new_node

		distances_end = scipy.spatial.distance_matrix(nodes, nodes)
		diff = abs(distances_end-distances)
		#print(diff, np.sum(diff)/2)
		print(iteration, moved, np.sum(diff)/2)
		if moved < update_thr:
			# The positions converged
			break


