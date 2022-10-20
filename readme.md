# Optical Flow Algorithms



# Time comparison:

| Algorithm 				| Seq.				| CPU Time (s) 	| Colab (CUDA) Time (s) |  
| - 						|-					|-				|-						|  
| Gunnar-Farneback			| val_seq_lq (100) 	| 1.76 			| 2.72					| 
| RAFT-s (iterative)		| val_seq_lq (100) 	| 60.77 		| -						| 
| RAFT-s (batch)			| val_seq_lq (100) 	| 77.37 		| 2.42					|
| RAFT (iterative)			| val_seq_lq (100) 	| 199.36		| -						| 
| RAFT (batch)				| val_seq_lq (100) 	| 247.15 		| 6.96					|
| SPyNet (iterative)		| val_seq_lq (100) 	| 32.17			| -						| 
| SPyNet (batch)			| val_seq_lq (100) 	| 34.35 		| 1.18					| 
