### Scraping

* Installer Scrapy et se placer dans le dossier
```
conda install scrapy
cd scraping/spiders
```

* Scraper les liens vers chaque discours et les stocker dans un JSON
```
scrapy runspider speechlinks_spider.py links.json
```

* Scraper les discours en entier via leurs liens
```
scrapy runspider speech_spider.py speeches.json
```

### Modèle

* Lancer le training (les arguments sont optionnels)
```
python network.py [-m <modele_depart.hdf5>] [--gpu]
```

* Générer un texte à partir d'un modèle (obligatoire)
```
python prediction.py <modele.hdf5> [-t <temp_sampling>]
```

* Changer l'architecture du modèle en modifiant network.py
* Changer les paramètres d'apprentissage et de prédiction en modifiant params.py
