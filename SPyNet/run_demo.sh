#!/bin/bash

echo "SINTEL"
python run.py --model=sintel-final --path=../demo-frames-sintel

echo "REDS-LQ-SEQ_1"
python run.py --model=sintel-final --path=../demo-frames-reds-lq/seq_1

echo "REDS-LQ-SEQ_2"
python run.py --model=sintel-final --path=../demo-frames-reds-lq/seq_2

echo "REDS-LQ-SEQ_3"
python run.py --model=sintel-final --path=../demo-frames-reds-lq/seq_3

echo "REDS-LONG-LQ-SEQ"
python run.py --model=sintel-final --path=../demo-frames-reds-lq/long_seq

echo "REDS-HQ-SEQ_1"
python run.py --model=sintel-final --path=../demo-frames-reds-hq/seq_1

echo "REDS-HQ-SEQ_2"
python run.py --model=sintel-final --path=../demo-frames-reds-hq/seq_2

echo "REDS-HQ-SEQ_3"
python run.py --model=sintel-final --path=../demo-frames-reds-hq/seq_3

echo "REDS-LONG-HQ-SEQ"
python run.py --model=sintel-final --path=../demo-frames-reds-hq/long_seq