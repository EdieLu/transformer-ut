#!/bin/bash

# evaluate bleu score

command="$0 $@"
cmddir=CMDs
echo "---------------------------------------------" >> $cmddir/eval_bleu.cmds
echo $command >> $cmddir/eval_bleu.cmds

model=en-de-v004
ckpt=35

# model=en-de-v011
# ckpt=38

fname=tst-COMMON #dev | tst-COMMON | tst-HE ｜ train_h1000
# tail=_bm10
tail=

DETOK=../lib/mustc-en-de-proc-fairseq/mosesdecoder/scripts/tokenizer/detokenizer.perl
fout=models/$model/eval_$fname$tail/$ckpt
bleu_scorer=./local/mosesdecoder/scripts/generic/multi-bleu-detok.perl

# -- bpe raw --
# refdir=../lib/mustc-en-de-proc/$fname/$fname.de.bpe50000
# $bleu_scorer $fout/translate.txt < $refdir > $fout/bleu.log

# -- bpe decoded --
# refdir=../lib/mustc-en-de-proc/$fname/$fname.de.dec
# python3 ./local/py-tools-v2/dec_bpe.py ../lib/mustc-en-de-proc/vocab/de-bpe-30000 $fout/translate.txt $fout/translate.txt.dec
# $bleu_scorer $fout/translate.txt.dec < $refdir > $fout/bleu.log.dec

# -- no bpe --
refdir=../lib/mustc-en-de/$fname/txt/$fname.de
$bleu_scorer $fout/translate.txt < $refdir > $fout/bleu.log

# -- fairseq bpe --
# sed -r 's/(@@ )|(@@ ?$)//g' $fout/translate.txt > $fout/translate.txt.rmbpe
# $DETOK -l de < $fout/translate.txt.rmbpe > $fout/translate.txt.rmbpe.detok
# refdir=../lib/mustc-en-de/$fname/txt/$fname.de
# $bleu_scorer $fout/translate.txt.rmbpe.detok < $refdir > $fout/bleu.log
