Nous vous souhaitons la bienvenue au centre du code ouvert de L’emploi en 2030 de l’Institut Brookfield! Ce code crée, analyse et teste le modèle qui a produit nos Prévisions sur la croissance des professions au Canada (PCPC).  

Nous vous fournissons ici le code utilisé pour créer le modèle, certaines des vérifications de modèle, et l’exercice d’analyse des éléments dont nous nous sommes servis pour déterminer les caractéristiques fondamentales, ainsi que les paires de caractéristiques mentionnées dans notre rapport. Ce code ne fournit pas le code R et les données pour l’analyse démographique (largement en raison de la taille des tableaux personnalisés de Statistique Canada), mais si vous avez la moindre question sur la méthode utilisée ou les mécanismes d’accès aux données, veuillez communiquer avec Josh à jzachariah@ryerson.ca. Nous vous conseillons vivement de lire le rapport (et son annexe) si vous ne l’avez pas déjà fait avant d’examiner le code. 

Anglais : *****Link
Français : ****** Link

Voici une description de chaque dossier et de son contenu. 

****raw_data****
Ce dossier renferme les données recueillies dans le cadre de nos six ateliers nationaux, ainsi que le score d’importance de la compétence O*NET pour chaque code de la classification nationale des professions (CNP). Nous vous rappelons que nous avions demandé aux participants aux ateliers d’évaluer chaque profession dans notre base d’apprentissage pour prédire si la part de l’emploi de ces professions allait augmenter, diminuer ou rester stable d’ici 2030.  

Les scores d’importance O*NET sont une mesure de 1 à 5 qui désigne l’importance d’une compétence, d’une connaissance ou d’une aptitude pour l’exécution du travail associé à une profession. Au total, il y a 120 compétences, caractéristiques de connaissance et aptitudes. Étant donné qu’O*NET est une base de données américaine, nous avons également inclus notre tableau de concordance entre les codes de profession américains et canadiens (SOC et CNP respectivement). 

Enfin, les fichiers de données d’O*NET et des ateliers renferment les descriptions des caractéristiques et ainsi que celles de la CNP respectivement, en anglais et en français. 

****tables****
model input: Les deux tableaux utilisés dans le modèle. Le fichier noc_answers renferme les réponses compilées de tous les participants à tous les ateliers, qui ont servi de base d’apprentissage. Le fichier noc_scores présente les scores d’importance des caractéristiques O*NET pour chaque profession. 

model output: Ce dossier contient nos projections issues des modèles d’augmentation et de diminution pour chaque profession. 

testing output: Dossier renfermant les résultats pour notre script de test (voir ci-dessous). 

sffs output: Ces deux fichiers texte énumèrent les caractéristiques retenues par notre processus de sélection de caractéristiques : Sequential Forward Floating Search.

feature analysis output: 
Le fichier 1run_non_conditional_influences présente les influences de chaque caractéristique non conditionnelle après une exécution de l’analyse. À noter que puisqu’il s’agit d’une seule exécution, des caractéristiques autres que les cinq caractéristiques fondamentales sont susceptibles d’atteindre le seuil de 95 %. Or, les cinq caractéristiques fondamentales sont les seules qui atteignent invariablement ce seuil, selon notre analyse de 10 exécutions du modèle. 
sig_pairs énumère toutes les importantes paires ordonnées de caractéristiques, avec leur part d’influence et d’occurrence.

****Final Model Scripts****
Ce dossier contient la version définitive de tous les scripts pertinents pour notre modèle.

**model_construction.ipynb**
Le script le plus important! Il crée et exécute les modèles d’augmentation et de diminution et produit les projections. 

**utils_rf.py**
Ensemble de fonctions utilisées pour un certain nombre des scripts pour la forêt aléatoire. BON NOMBRE de ces fonctions sont utilisées pour les scripts dans «?old files?» (vieux fichiers). Une description des fonctions les plus importantes est donnée ci-dessous.

data_process(file,discrete)
Cette fonction traite les données. Le drapeau discret permet à l’utilisateur de faire arrondir ou non les scores O*NET. Nous arrondissons les scores pour nos modèles définitifs. 

init_params(model_type)
Détermine le paramètre pour le modèle de forêt aléatoire. Le modèle final utilise le paramètre cat. 

run_k_fold(x,y,params,index,binned,model_type)
Cette fonction exécute la méthode group k-fold que nous avons utilisée pour tester notre modèle. .

param_search(x,y,model_type)
Nous avons utilisé cette fonction pour trouver les paramètres optimaux (à nouveau, en utilisant la méthode group k fold). Nous avons utilisé diverses grilles de paramètres qui ne sont pas présentés dans le présent dossier, ainsi qu’un processus itératif pour préciser la zone délimitée par les paramètres. 

run_sfs(x,y,model_type,custom_score,increase_model)
Exécute la recherche de caractéristiques SFFS. Le paramètre custom_score détermine si nous utilisons ou non notre fonction personnalisée d’erreur absolue moyenne (EAM) pour évaluer les ensembles de caractéristiques (voir ci-dessous). 

custom_mae_increase(y_true,y_pred) and custom_mae_decrease(y_true,y_pred)
Notre modèle apprend en fonction des données au niveau du participant, mais effectue les tests en vérifiant l’EAM pour la part prédite contre la part réelle des experts qui donnent une réponse. Ces fonctions effectuent cette agrégation et calculent le score.  

**testing**
Ces scripts exécutent les diverses méthodes de test utilisées dans l’annexe du rapport. Le script regional_models.ipynb teste la mesure dans laquelle le modèle ayant appris d’un atelier est capable de prédire les réponses des autres. 

**sffs****
Ces deux scripts exécutent le modèle SFFS 20 fois et inscrivent les résultats dans un fichier pickle. Si vous les exécutez, nous vous recommandons VIVEMENT de faire appel à un service d’infonuagique. L’exécution des scripts demande beaucoup de temps, mais ces derniers sont configurés de façon à utiliser autant de fils que vous leur donnez, ce qui, en raison de la nature du processus, réduit sensiblement le temps d’exécution. 

**Feature Analysis Files**
Le script dans ce dossier exécute la méthode définitive utilisée pour déterminer l’influence d’une caractéristique ou d’une paire de caractéristiques. D’autres méthodes et tentatives se trouvent dans old_files. Des détails sur le processus sont exposés dans notre annexe. 

*find_trait_influences.ipynb*
Il s’agit du script utilisé pour notre exercice d’analyse des éléments. Il donne toutes les influences de caractéristiques ainsi que les paires de caractéristiques importantes. Voir notre annexe pour obtenir plus de détails. 

*basic feature analysis*
Ce script contient de nombreux tests pour découvrir une panoplie d’aspects que vous souhaiteriez connaître au sujet des caractéristiques. 

****old files****
Ce dossier stocke un éventail d’autres approches, tests et modèles que nous avons essayés. Nous vous invitons à l’explorer, mais gardez à l’esprit qu’ils ne fonctionnent pas nécessairement tous. 

