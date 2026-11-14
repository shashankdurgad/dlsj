#!/bin/bash
# export_ckpt.sh CKPT NAME : one RTC trace + INT8 (no optimisation sweep), into ~/inference/exports/molmo_opt/NAME
set -e
CKPT=$1; NAME=$2; OUT=~/inference/exports/molmo_opt/$NAME
case "$CKPT" in *.partial|*.crdownload|*.download) echo "$CKPT is an unfinished copy"; exit 1;; esac
prev=-1; while cur=$(stat -c %s "$CKPT") && [ "$cur" != "$prev" ]; do [ "$prev" -ge 0 ] && echo "waiting: $((cur/1000000)) MB"; prev=$cur; sleep 20; done
[ "$cur" -gt 15000000000 ] || { echo "$CKPT is only $((cur/1000000)) MB: incomplete?"; exit 1; }
export HF_HOME=~/inference/hf-molmo HF_HUB_OFFLINE=1 TMPDIR=~/inference/tmp/onnx
mkdir -p $OUT $TMPDIR; cd ~/inference/physical-ai-studio/library
echo "[$(date +%H:%M:%S)] trace rtc_fp16 from $CKPT"
.venv/bin/python ~/inference/deploy/molmoact2/export.py --ckpt "$CKPT" --rtc --out $OUT/rtc_fp16
echo "[$(date +%H:%M:%S)] quantize rtc_int8"
.venv/bin/python ~/inference/deploy/molmoact2/quantize.py $OUT/rtc_fp16 $OUT/rtc_int8 --variant int8_sym
echo "[$(date +%H:%M:%S)] DONE $OUT/rtc_int8"
