#!/bin/bash

# evaluate bleu score

command="$0 $@"
cmddir=CMDs
echo "---------------------------------------------" >> $cmddir/eval_bleu.cmds
echo $command >> $cmddir/eval_bleu.cmds

# ---------- [model] -------------
model=gec-v031
ckpt=19
libbase=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe
tail=
# ---------- [files] -------------

# fname=test_fce_test
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-clc/fce-test

# fname=test_clc
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-clc/clc

# fname=test_nict
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-nict/nict

# fname=test_nict_new
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-nict-new/nict

# fname=test_dtal
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-dtal/dtal

fname=test_dtal_asr # default seg manual
ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-dtal-asr/dtal-asr

# ---------- [score] -------------
gleuscorer=./local/gleu/compute_gleu.py
outdir=models/$model/$fname/$ckpt
srcdir=$ftst.src$tail
refdir=$ftst.tgt$tail

python $gleuscorer -r $refdir -s $srcdir -o $outdir/translate.txt > $outdir/gleu.log &
