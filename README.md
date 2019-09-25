
# fifty_shades_of_ml

Ce repository contient le projet de Reinforcement Learning appliqué à la gestion de portefeuilles.

## Description

L'idée du projet et d'explorer le Reinforcement Learning pour l'appliquer au problème spécifique de la gestion de portefeuille chez Aequam Capital.
Il y a plusieurs étapes dans ce projet :

* **aequam_env.py** permet de créer un environnement de Reinforcement Learning sur le format d'OpenAI Gym (la classe AequamEnv hérite de gym.Env) ;
* **Aequam_env.ipynb** utilise le module ci-dessus pour créer un modèle de Reinforcement Learning à vocation prédictive (prédiction d'une allocation de portefeuille). Ce modèle a de nombreux paramètres qui peuvent être testés à l'aveugle ou plutôt optimisés grâce à scikit-optimize.

Ce projet n'est pas terminé et de nombreuses extensions peuvent et doivent être mises en place avant mise en production. Elles sont décrites dans les documentations pdf attachées au projet (dans le dossier "documentation").


## Prise en main

### Installation de l'environnement de code

Au choix : 
* Si vous utilisez Anaconda, vous pouvez installer les librairies requises en important directement l'environnement du projet :
```
conda env create -f environment.yml
```
* Sinon ou si vous préférez ne pas créer d'environnement, vous pouvez utiliser le fichier requirements.txt et la commande pip :
```
pip install -r requirements.txt
```

### Mise en place

Une fois l'environnement créé ou les librairies installées, tout est prêt à l'emploi, bien penser simplement à dézipper **results_we.pkl** si vous voulez utiliser ce fichier de résultat dans le notebook **Aequam_env.ipynb**.

On peut également ajouter les dossiers **reports** et **tmp** au fichier **.gitignore** pour ne pas trop alourdir le repository.

### Description du contenu et utilisation

* Les fichiers **.gitignore**, **environment.yml**, **requirements.txt** et **README.md** sont standards ;
* Le dossier **data** contient les jeux de données brutes et transformées qui sont utilisés dans **Aequam_env.ipynb**. Ces jeux de données sont expliqués et peuvent être modifiés dans le notebook **Create and visualize datasets.ipynb** qui leur est dédié ;
* Le dossier **documentation** contient de la documentation sur le Reinforcement Learning en général et sur le projet en particulier. Les fichiers à l'intérieur sont numéroter pour une appréhension logique et progressive du projet ;
* Le dossier **reports** contient des résultats de backtests de modèles de Reinforcement Learning. Ces backtests sont lancés depuis le notebook **Aequam_env.ipynb**. Le dossier contient déjà des fichiers à titre d'exemple mais on peut en créer autant qu'on veut. Une fois le repository git cloné, il est conseillé d'ajouter ce dossier au **.gitignore** car il peut vite devenir volumineux ;
* Le dossier **tmp** contient des fichiers de log TensorBoard qui sont lancés quand un modèle de Reinforcement Learning est en phase d'apprentissage. Là encore, il contient un fichier de log à titre d'exemple mais de nombreux sont créés automatiquement pendant l'optimisation. Une fois le repository git cloné, il est conseillé d'ajouter ce dossier au **.gitignore** car il peut vite devenir volumineux ;
* Le notebook **ACP.ipynb** contient une implémentation de l'Analyse en Composantes Principales sur un jeu de données, de manière "roulante" (sans forward-looking). Il permet de recréer cette ACP sur n'importe quel jeu de données et d'en analyser les résultats. On peut l'utiliser dans une phase amont (création de nouvelles variables explicatives) ;
* Le fichier **backtest_utils.py** contient un certain nombre de fonctions basiques utiles à droite à gauche qu'on stocke ici pour alléger la lecture des différents codes. Ce module est appelé dans presque tous les notebooks ;
* Le fichier **make_dataset.py** est en réalité peu utile. Il a servi originellement à construire des jeux de données utilisant des indicateurs d'analyse technique. On le laisse sur ce repository, à toutes fins utiles pour celui qui voudra utiliser ces indicateurs pour créer de nouveaux jeux de données ;
* Le fichier zippé **results_we.pkl** contient la sortie d'une optimisation via scikit-optimize qui est un peu longue et qui a été sauvegardée à des fins d'exemple. Il faut le dézipper pour pouvoir l'utiliser.
* Les deux fichiers les plus importants sont **aequam_env.py** et **Aequam_env.ipynb**. Ils sont présentés dans la description et sont abondamment commentés. 

## Aide

Ne pas hésiter à contacter l'auteur en cas de besoin.

## Auteur

Dorian LAGADEC
dorian.lagadec@hec.edu

## Acknowledgments

Les articles et liens utiles sont tous dans la documentation.