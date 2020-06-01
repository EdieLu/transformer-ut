#!/bin/bash

# evaluate bleu score

command="$0 $@"
cmddir=CMDs
echo "---------------------------------------------" >> $cmddir/eval_bleu.cmds
echo $command >> $cmddir/eval_bleu.cmds

fname=dev #dev | tst-COMMON | tst-HE
model=en-de-v001

for i in `seq 1 1 31`
do
    echo $i
    ckpt=$i
    fout=models/$model/eval_$fname/$ckpt

    # -- bpe raw --
    # refdir=../lib/mustc-en-de-proc/$fname/$fname.de.bpe50000
    # python ./local/py-tools/bleu_scorer.py $fout/translate.txt $refdir > $fout/bleu.log

    # -- bpe decoded --
    refdir=../lib/mustc-en-de-proc/$fname/$fname.de.dec
    python3 ./local/py-tools-v2/dec_bpe.py ../lib/mustc-en-de-proc/vocab/de-bpe-30000 $fout/translate.txt $fout/translate.txt.dec
    python ./local/py-tools/bleu_scorer.py $fout/translate.txt.dec $refdir > $fout/bleu.log.dec

    # -- no bpe --
    # refdir=../lib/mustc-en-de-proc/$fname/$fname.de.dec
    # python ./local/py-tools/bleu_scorer.py $fout/translate.txt $refdir > $fout/bleu.log

done
