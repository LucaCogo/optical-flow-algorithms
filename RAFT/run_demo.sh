#!/bin/bash

echo "SINTEL"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-sintel --output_folder=raft
 

echo "REDS-LQ-SEQ_1"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-lq/seq_1 --output_folder=raft

echo "REDS-LQ-SEQ_2"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-lq/seq_2 --output_folder=raft

echo "REDS-LQ-SEQ_3"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-lq/seq_3 --output_folder=raft


echo "REDS-LONG-LQ-SEQ"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-lq/long_seq --output_folder=raft


echo "REDS-HQ-SEQ_1"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-hq/seq_1 --output_folder=raft


echo "REDS-HQ-SEQ_2"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-hq/seq_2 --output_folder=raft


echo "REDS-HQ-SEQ_3"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-hq/seq_3 --output_folder=raft


echo "REDS-LONG-HQ-SEQ"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-hq/long_seq --output_folder=raft




echo "SINTEL-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-sintel --small --output_folder=raft-s
 

echo "REDS-LQ-SEQ_1-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-lq/seq_1 --small --output_folder=raft-s

echo "REDS-LQ-SEQ_2-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-lq/seq_2 --small --output_folder=raft-s


echo "REDS-LQ-SEQ_3-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-lq/seq_3 --small --output_folder=raft-s


echo "REDS-LONG-LQ-SEQ-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-lq/long_seq --small --output_folder=raft-s


echo "REDS-HQ-SEQ_1-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-hq/seq_1 --small --output_folder=raft-s


echo "REDS-HQ-SEQ_2-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-hq/seq_2 --small --output_folder=raft-s


echo "REDS-HQ-SEQ_3-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-hq/seq_3 --small --output_folder=raft-s


echo "REDS-LONG-HQ-SEQ-small"
python demo.py --model=models/raft-small.pth --path=../demo-frames-reds-hq/long_seq --small --output_folder=raft-s




