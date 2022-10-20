# Optical Flow Algorithms



# Time comparison:

| Algorithm 				| Seq.				| CPU Time (s) 	| Colab (CUDA) Time (s) |  
| - 						|-					|-				|-						|  
| Gunnar-Farneback			| val_seq_lq (100) 	| 1.76 			| 2.72					| 
| RAFT-s (iterative)		| val_seq_lq (100) 	| 40.52 		| -						| 
| RAFT-s (batch)			| val_seq_lq (100) 	| 51.59 		| 2.42					|
| RAFT (iterative)			| val_seq_lq (100) 	| 133.87		| -						| 
| RAFT (batch)				| val_seq_lq (100) 	| 167.89 		| 6.96					|
| SPyNet (iterative)		| val_seq_lq (100) 	| 35.21			| -						| 
| SPyNet (batch)			| val_seq_lq (100) 	| 34.79 		| 1.18					| 




