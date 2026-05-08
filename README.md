# it3

primero que nada

    source venv/bin/activate
    pip install -r requirements.txt

## SSLIM

    python -m src.main --modelo SSLIM --max_jsons 1 --max_playlists 400 --train-dir data/dataset/train --eval-dir data/dataset/eval/ --lr 0.01  --epochs 30


## FISM

    python -m src.main --modelo FISM --max_jsons 1 --max_playlists 400 --train-dir data/dataset/train --eval-dir data/dataset/eval/ --dim 30 --lr 0.01  --epochs 30

aparte del modelo la única diferencia es que a FISM hay que especificarle `dim`, que viene a ser:

 $S_{n\times n} = Q_{n \times dim} \times T_{dim \times n}$
