#!/bin/bash

# evaluate bleu score

command="$0 $@"
cmddir=CMDs
echo "---------------------------------------------" >> $cmddir/eval_bleu.cmds
echo $command >> $cmddir/eval_bleu.cmds

fname=tst-COMMON #dev | tst-COMMON | tst-HE | tst_iwslt17
model=en-de-v008

DETOK=../lib/mustc-en-de-proc-fairseq/mosesdecoder/scripts/tokenizer/detokenizer.perl
bleu_scorer=./local/mosesdecoder/scripts/generic/multi-bleu-detok.perl

for i in `seq 1 1 50`
do
    echo $i
    ckpt=$i
    fout=./models/$model/eval_$fname/$ckpt

    # -- bpe raw --
    # refdir=../lib/mustc-en-de-proc/$fname/$fname.de.bpe50000
    # $bleu_scorer $fout/translate.txt < $refdir > $fout/bleu.log

    # -- bpe decoded --
    # refdir=../lib/mustc-en-de-proc/$fname/$fname.de.dec
    # python3 ./local/py-tools-v2/dec_bpe.py ../lib/mustc-en-de-proc/vocab/de-bpe-30000 $fout/translate.txt $fout/translate.txt.dec
    # $bleu_scorer $fout/translate.txt.dec < $refdir > $fout/bleu.log.dec

    # -- no bpe --
    # refdir=../lib/mustc-en-de/$fname/txt/$fname.de
    # $bleu_scorer $fout/translate.txt < $refdir > $fout/bleu.log

    # -- fairseq bpe --
    sed -r 's/(@@ )|(@@ ?$)//g' $fout/translate.txt > $fout/translate.txt.rmbpe
    $DETOK -l de < $fout/translate.txt.rmbpe > $fout/translate.txt.rmbpe.detok
    refdir=../lib/mustc-en-de/$fname/txt/$fname.de
    # refdir=../lib/wmt17_en_de/wmt17_en_de/tmp/test.de
    $bleu_scorer $fout/translate.txt.rmbpe.detok < $refdir > $fout/bleu.log

done
