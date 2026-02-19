# :)

```
data
├── dataset
│   ├── eval
│   │   ├── test_eval_playlists.json
│   │   └── test_input_playlists.json
│   └── train
│       ├── mpd.slice.0-999.json
│       ├── mpd.slice.1000-1999.json
│       ├── mpd.slice.10000-10999.json
│       ├── mpd.slice.100000-100999.json
│       ├── mpd.slice.995000-995999.json
│       ├── mpd.slice.996000-996999.json
│       ├── mpd.slice.997000-997999.json
│       ├── mpd.slice.998000-998999.json
│       └── mpd.slice.999000-999999.json
├── license.txt
├── matrix_meta.pkl
├── matrix.npz
├── md5sums
├── README.md
├── src
│   ├── check.py
│   ├── deeper_stats.py
│   ├── descriptions.py
│   ├── print.py
│   ├── show.py
│   └── stats.py
└── stats.txt
```



# TODO:
 - metricas separadas y miradas bien https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
 - limpiar cosas que no son de uso evidente a utils
 - comentarios y docstrings justos y necesarios
 - exportar metricas a un `.tex` con un `createtable`
 - revisar flujo general
 - formato consistente en sparse dense y tipo de sparse. (documentar)
