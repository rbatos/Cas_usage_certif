# Sujet d'Examen : Orientation et tri multimodal des demandeurs d'emploi par l'Intelligence Artificielle

L'optimisation des politiques publiques de l'emploi et le développement des outils d'aide à la décision imposent des solutions d'aiguillage rapides et fiables. Ce projet consiste à concevoir un système d'intelligence artificielle capable de classer le délai potentiel de retour à l'emploi d'un usager à partir de données hybrides (numériques, catégorielles et textuelles). 

Pour créer ce système d'IA, vous aurez à votre disposition un jeu de données composé de données collectées au sein d'une agence nationale de l'emploi. Il contient 2 500 échantillons pour lesquels nous connaissons la classe réelle de retour à l'emploi (c'est la décision de parcours ou le constat terrain validé par les conseillers). 

Dans le détail, chaque échantillon est composé de :
- Données tabulaires : Âge, données administratives, critères socio-professionnels (niveau de diplôme, ancienneté dans le dernier poste, statut d'allocation), variable sensible (nationalité hors UE) et critères géographiques (code INSEE de la commune). 
- Données textuelles : Synthèse libre rédigée par le conseiller à la suite du premier entretien d'aiguillage, détaillant le projet et les freins périphériques. 
- Variable cible : Classe de retour à l'emploi (0 : Rapide < 6 mois, 1 : Moyen entre 6 et 12 mois, 2 : Risque de longue durée > 12 mois). 

L'enjeu pour vous est de réussir à exploiter ces données pour prédire si une situation rencontrée relève d'un risque d'inactivité de longue durée, d'un délai moyen, ou au contraire d'une reprise d'activité rapide. Vous devrez pour cela respecter les contraintes réglementaires et éthiques liées à l'usage de données personnelles. Vous devrez aussi adapter vos apprentissages afin de limiter le nombre de mauvaises classifications dangereuses d'un point de vue métier. 

## Objectif du projet

L'objectif est de prédire la classe de retour à l'emploi d'une demande entrante, à partir des informations renseignées. Cette tâche est donc une tâche supervisée de classification à 3 classes. 

Vous devrez :
- Entraîner plusieurs modèles (Random Forest, LightGBM, XGBoost, etc.) et comparer leurs performances. 
- Utiliser des métriques de classification appropriées : Accuracy, F1-score macro et Matrice de confusion. 
- Mettre en place une validation croisée stratifiée pour garantir la robustesse des résultats face au déséquilibre des classes. 
- Gérer le nettoyage et le préprocessing complet : Imputation des valeurs manquantes présentes dans le fichier et ingénierie de caractéristiques (Feature Engineering) sur les variables à haute cardinalité (notamment l'extraction du département à partir du code INSEE).
- Discuter de l'architecture de modèle qui présente le meilleur rapport performance / coût computationnel d'inférence. 

Une fois cette tâche effectuée, vous devrez entamer une réflexion sur un enjeu éthique et métier majeur de votre cas d'usage : **« Une situation d'un niveau de risque 2 (chômage de longue durée) classée niveau 0 (retour rapide) est une erreur bien plus grave d'un point de vue humain et financier qu'une situation de niveau 0 classée niveau 1 ou 2, car elle prive l'usager d'un accompagnement renforcé indispensable. »**

Concrètement, il vous sera demandé de modifier votre meilleur modèle pour favoriser la diminution des erreurs critiques. Pour cela, vous devrez trouver quels paramètres ou quelles métriques favoriser lors de l'apprentissage (poids des classes, seuils de décision). 

## Analyse comparée des scénarios

Un point essentiel de votre mission sera de comparer les performances obtenues selon différents scénarios d'entraînement, afin d'estimer l'impact des données sensibles ou des différents types de données disponibles : 
- **Scénario 1 — Approche multimodale complète** : Utilisation de l'intégralité des variables (tabulaires + texte vectorisé). 
- **Scénario 2 — Sans variables sensibles (Approche Éthique)** : Retrait des variables considérées comme éthiquement sensibles avec justification argumentée au sens de la CNIL et du RGPD. 
- **Scénario 3 — Diagnostic par le texte seul (NLP pure)** : Prédiction basée exclusivement sur la synthèse écrite de l'entretien de cadrage pour tester la robustesse du modèle de langage face à des verbatims contenant du bruit stochastique. 
- **Scénario 4 — Données contextuelles pures (Tabulaire seul)** : Prédicteur basé uniquement sur l'âge, les diplômes, l'ancienneté et la géographie, sans l'apport du contexte sémantique textuel. 

Votre objectif est de documenter et commenter l'impact de chaque scénario sur les performances des modèles, et d'argumenter sur le choix du modèle et des données à utiliser dans un contexte réel, en tenant compte des enjeux éthiques, légaux et de robustesse. 


## Industrialisation et Déploiement

Une fois les modèles de prédiction sélectionnés et validés, vous passerez à la phase d'industrialisation de la solution IA. Vous devez contribuer à la conception et à l'évaluation de la proposition d'architecture cible pour intégrer cette brique prédictive au sein du Système d'Information (SI) de l'agence.

Cette conception doit impérativement prendre en compte les contraintes réelles suivantes :
- **Contraintes d'intégration et de flux** : L'API ne doit pas fonctionner en vase clos. Vous devez cartographier les flux de données entre l'application de guichet unique des conseillers (qui consomme l'inférence via HTTP POST), le référentiel national des usagers (base de données transactionnelle) et votre service d'IA.
- **Contraintes de performance et d'infrastructure** : Le modèle multimodal combine du texte (TF-IDF/Embeddings) et des variables tabulaires encodées. Vous évaluerez le rapport performance/coût computationnel d'inférence et discuterez du choix de l'infrastructure d'hébergement : déploiement sur site (on-premise) pour des raisons de confidentialité des données publiques ou sur un Cloud souverain. Vous devrez estimer la latence acceptable pour un conseiller en entretien de face-à-face et dimensionner les ressources (CPU/RAM). 

L'API permettra de mettre à disposition le modèle d'IA sous forme de service, que l'on pourra interroger via des requêtes HTTP POST. Elle devra inclure un mécanisme de chargement du modèle entraîné, et exposer au minimum une route pour prédire le niveau de risque d'une situation à partir des caractéristiques fournies, une route de réentraînement monitoré pour intégrer le feedback correctif des conseillers, et une route pour mesurer la santé de votre API (health check). 

Vous veillerez à journaliser chaque requête, en sauvegardant les entrées, les sorties, les dates d'inférence, et idéalement l'ID de session. Une attention particulière sera portée à la gestion des erreurs, à la validation des données entrantes, et à la robustesse de l'API. 

L'interface graphique devra permettre à un conseiller de renseigner les informations liées à un usager via un formulaire simple, d'obtenir une prédiction claire, et de consulter un historique des inférences. 

Par ailleurs, vous intégrerez des outils de suivi de modèle avec MLflow pour tracer les versions du modèle, ses hyperparamètres, ses performances, et permettre un versioning contrôlé. Enfin, un monitoring de l'API pourra être mis en place à l'aide de solutions comme Prometheus + Grafana ou simplement via des logs et alertes automatisées. 

Le projet devra intégrer une démarche CI/CD complète, en utilisant GitHub Actions pour automatiser les étapes clés du cycle de vie du code : tests unitaires, linting, entraînement et déploiement. À chaque mise à jour de la branche principale, une image Docker de l'API sera construite automatiquement et déployée sur l'infrastructure cible, garantissant un déploiement reproductible, traçable et conforme aux bonnes pratiques d'ingénierie IA. 

Cette phase vise à valider votre capacité à transformer un modèle de data science en une solution exploitable en environnement réel, dans le respect des contraintes de fiabilité, d'éthique et de gouvernance des données. 

## Éthique et Cadre Réglementaire

Une section du rapport de projet devra traiter spécifiquement de la conformité au RGPD, de la non-discrimination algorithmique des profils (Loi pour une République Numérique), et de la responsabilité juridique de l'éditeur ou de l'administration publique en cas d'erreur d'orientation induisant une perte de chance pour l'usager. 

## Description des données disponibles

| Colonne | Description |
| - | - |
| usager_id | Identifiant unique du demandeur d'emploi |
| age | Âge de l'usager en années |
| niveau_diplome | Plus haut niveau de diplôme obtenu (Sans diplôme, Bac, Bac+2, Bac+5) (Donnée manquante potentielle) |
| anciennete_poste_ans | Nombre d'années d'expérience dans le dernier emploi occupé |
| code_rome_vise | Code à 5 caractères du Répertoire Opérationnel des Métiers et des Emplois ciblé par l'usager |
| code_insee_commune | Code officiel géographique INSEE de la commune de résidence  |
| est_allocataire | Statut d'indemnisation de l'usager : 1 (Oui), 0 (Non) |
| nationalite_hors_ue | Variable sensible : 1 (Nationalité hors Union Européenne), 0 (Nationalité UE) |
| synthese_entretien | Texte libre contenant les notes textuelles du conseiller lors de l'entretien  |
| classe_retour_emploi | Variable cible : 0 (Rapide < 6m), 1 (Moyen 6-12m), 2 (Risque de longue durée > 12m) |

