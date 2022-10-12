#!/bin/bash

echo "SINTEL"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-sintel

echo "REDS-LQ-SEQ_1"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-lq/seq_1

echo "REDS-LQ-SEQ_2"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-lq/seq_2

echo "REDS-LQ-SEQ_3"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-lq/seq_3

echo "REDS-HQ-SEQ_1"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-hq/seq_1

echo "REDS-HQ-SEQ_2"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-hq/seq_2

echo "REDS-HQ-SEQ_3"
python demo.py --model=models/raft-sintel.pth --path=../demo-frames-reds-hq/seq_3


