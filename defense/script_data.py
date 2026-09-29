# -*- coding: utf-8 -*-
"""Texte oral de la soutenance (45 minutes) et questions-réponses du jury.
Source unique : sert à produire le PDF du script et les notes du PowerPoint.
Tous les chiffres proviennent de la thèse (version v29)."""

# (titre de la diapositive, durée en secondes, texte oral)
SLIDES = [
("Page de titre", 60,
"Monsieur le Président, Mesdames et Messieurs les membres du jury, bonjour. Je vous remercie d'avoir accepté "
"d'évaluer ce travail. Je vais vous présenter ma thèse intitulée « Deep Learning-Based Prediction of Disease "
"Severity Using Medical Imaging: An Application to Diabetology ». En une phrase : cette thèse étudie comment trois "
"types d'information – les relations entre les images, la topologie des vaisseaux de la rétine et l'ordre des grades "
"de la maladie – peuvent améliorer le classement automatique de la rétinopathie diabétique. Les trois cercles R, S et O "
"que vous voyez reviendront tout au long de la présentation : R pour relationnel, S pour structurel, c'est-à-dire la "
"topologie, et O pour ordinal."),

("Plan", 45,
"La présentation suit la logique de la thèse. Je commence par le contexte médical, que j'expliquerai simplement. "
"Je présente ensuite le problème, les trois hypothèses et la méthodologie, puis les outils utilisés et les premiers "
"résultats qui ont motivé le travail. Viennent ensuite les trois systèmes : TOPORET, puis DGTS et DOTS. Je dirai un "
"mot de la partie logicielle et réglementaire, et je terminerai par une synthèse : ce qui est établi, ce qui ne l'est "
"pas encore, et les perspectives."),

("Partie 1 – Contexte et problème", 15,
"Première partie : le contexte, et pourquoi le classement de la rétinopathie diabétique est un problème de sécurité, "
"pas seulement de précision."),

("La rétinopathie diabétique : une cause de cécité silencieuse et évitable", 75,
"Le diabète abîme progressivement les petits vaisseaux sanguins du corps, et en particulier ceux de la rétine, au fond "
"de l'œil. C'est la rétinopathie diabétique. Environ 537 millions d'adultes vivent avec un diabète, et l'on en attend "
"783 millions en 2045. Environ 103 millions de personnes ont une rétinopathie diabétique, et c'est l'une des premières "
"causes de cécité évitable chez l'adulte en âge de travailler. La difficulté principale est que la maladie est "
"silencieuse : le patient voit normalement jusqu'à un stade avancé, alors que le traitement est surtout efficace tôt. "
"La seule réponse efficace à l'échelle d'une population est donc le dépistage régulier : on photographie le fond d'œil "
"de chaque patient diabétique, et un expert classe l'image. Or le nombre d'images dépasse aujourd'hui largement le "
"nombre d'experts disponibles, surtout là où les ophtalmologistes sont rares. Le classement automatique par "
"apprentissage profond est donc la réponse naturelle."),

("Les cinq grades ICDR", 90,
"Voici ce que voit l'expert. L'échelle internationale ICDR compte cinq grades. Le grade 0 : pas de rétinopathie. Le "
"grade 1, légère : seulement des micro-anévrismes, de petites dilatations des capillaires. Le grade 2, modérée : des "
"hémorragies et des exsudats apparaissent. Le grade 3, sévère, défini par la règle dite 4-2-1 : de nombreuses "
"hémorragies dans les quatre quadrants, ou des veines en chapelet, ou des anomalies microvasculaires. Enfin le grade "
"4, proliférant : de nouveaux vaisseaux anormaux se forment ; c'est une urgence. Deux faits guident toute la thèse. "
"Premièrement, les grades sont ordonnés : confondre un grade 0 avec un grade 4 est beaucoup plus grave que confondre un "
"grade 1 avec un grade 2. Deuxièmement, les grades 3 et 4 doivent être orientés vers l'ophtalmologiste en quelques "
"semaines, voire en urgence. Ce sont donc les erreurs sur ces grades qui comptent le plus pour le patient."),

("Pourquoi la précision ne suffit pas", 90,
"Un système peut avoir une très bonne précision globale et pourtant laisser passer des cas graves. Pour le mesurer, "
"j'utilise le Sev-NR, le taux de non-orientation des cas sévères : la proportion d'images de grade 3 que le système "
"classe « pas de rétinopathie » ou « légère », et qui ne seraient donc pas orientées. Le PDR-NR est la même mesure pour "
"le grade 4. Un exemple illustratif : un programme qui dépiste 500 000 patients par an, dont environ 25 000 ont une "
"forme sévère. Avec un Sev-NR de 5,8 % – c'est la valeur d'un CNN de référence sur la base externe Messidor-2 dans "
"cette thèse –, environ 1 450 cas sévères ne seraient pas orientés chaque année. Je précise que c'est une projection "
"sous hypothèses de prévalence, pas un résultat clinique. Mais elle explique un choix de la thèse : les métriques de "
"sécurité sont toujours rapportées à côté de la précision, et vérifiées sur des données d'un autre pays."),

("Trois limites des systèmes représentatifs", 90,
"En étudiant les systèmes représentatifs de la littérature – par exemple Gulshan et al. en 2016, CABNet en 2021 ou "
"RETFound en 2023 –, on observe trois limites d'architecture. L1 : chaque image est classée isolément ; des patients "
"qui se ressemblent, et qui ont souvent le même grade, ne s'informent jamais. L2 : les réseaux de convolution voient la "
"texture locale, mais ne mesurent pas explicitement la structure globale de l'arbre vasculaire – ses composantes et ses "
"boucles –, qui change avec la maladie. L3 : les grades sont traités comme des catégories sans ordre, et le taux de cas "
"sévères non orientés est rarement mesuré, encore moins sur des données externes. Je souligne que c'est une remarque "
"d'architecture et non un jugement sur toute la littérature : plusieurs systèmes ont été validés prospectivement et "
"déployés."),

("Partie 2 – Hypothèses et méthodologie", 15,
"Deuxième partie : ce que la thèse cherche à tester, et comment."),

("Objectif et hypothèses de recherche", 90,
"L'objectif est unique : exploiter les relations, la topologie et l'ordinalité pour améliorer le classement de la "
"rétinopathie diabétique. Il se traduit par trois hypothèses, chacune liée à une limite. H1, l'hypothèse relationnelle : "
"un graphe de population apporte une information au-delà de celle d'un classifieur image par image. H2, l'hypothèse "
"structurelle : l'homologie persistante du réseau vasculaire est statistiquement associée à la sévérité. H3, l'hypothèse "
"de sécurité ordinale : une chaîne ordinale satisfait des objectifs de dépistage fixés à l'avance – un QWK d'au moins "
"0,80, des taux de non-orientation d'au plus 2 %, et aucune confusion entre les grades 0 et 4 –, en interne et sur des "
"données externes ; c'est H3a. Et aucune de ses variantes réduites n'y parvient ; c'est H3b. Pour H1 et H2, le test, le "
"seuil de décision et la mesure de taille d'effet ont été fixés avant l'expérience, pour qu'un résultat négatif ne "
"puisse pas être expliqué après coup."),

("Un cadre de recherche à trois axes", 90,
"Un choix méthodologique essentiel : la thèse propose un cadre de recherche à trois axes, et non un modèle unique. "
"Chaque axe est porté par un système dédié, pour pouvoir mesurer sa contribution séparément, avec sa propre ablation. "
"TOPORET teste ensemble les relations et la topologie des vaisseaux, sur trois bases publiques, selon un protocole "
"transductif ; il porte H1 et H2. DGTS reprend ces deux ingrédients dans un graphe inductif, avec un régulariseur "
"topologique dans l'espace des caractéristiques, et il est évalué sans réentraînement sur EyePACS. DOTS ajoute "
"l'ordinalité et est évalué contre les objectifs de sécurité sur Messidor-2 ; il porte H3. Ces trois systèmes ne "
"fournissent donc pas la même forme de preuve, et je le dis explicitement : preuve transductive pour TOPORET, preuve "
"inductive et externe pour DGTS, preuve orientée dépistage, en estimation ponctuelle, pour DOTS."),

("Un cadre commun en six étapes", 60,
"Tous les systèmes reposent sur une chaîne en six étapes, avec des interfaces typées : le prétraitement et l'extraction "
"du squelette vasculaire, un réseau de convolution, le calcul de l'homologie persistante, la fusion des "
"caractéristiques, le graphe de population, et enfin la tête de classement. Les couleurs indiquent l'axe auquel "
"appartient chaque étape. Ces interfaces typées permettent de tester chaque étape séparément ; je m'en sers au "
"chapitre 6 pour la vérification logicielle."),

("Partie 3 – Outils et premières preuves", 15,
"Troisième partie : les trois outils, puis les premiers résultats qui ont conduit à TOPORET."),

("Trois outils expliqués simplement", 105,
"Trois outils, expliqués simplement. Le premier est le graphe de population, traité par GraphSAGE. Chaque image est un "
"nœud ; on relie les images qui se ressemblent ; puis le réseau agrège, pour chaque nœud, l'information de ses voisins. "
"L'idée est qu'un cas ressemble souvent à ses voisins. Le deuxième est l'homologie persistante. On considère le "
"squelette des vaisseaux comme un nuage de points, et l'on fait grossir progressivement une boule autour de chaque "
"point. À chaque échelle, on compte les composantes connexes, notées H0, et les boucles, notées H1. Une boucle qui "
"persiste sur une large gamme d'échelles est une vraie structure ; une boucle qui disparaît aussitôt est du bruit. Le "
"troisième est la régression ordinale CORAL. Au lieu de choisir parmi cinq classes sans ordre, le modèle répond à quatre "
"questions binaires emboîtées : le grade est-il au moins 1, au moins 2, au moins 3, au moins 4 ? L'ordre des grades est "
"ainsi inscrit dans le modèle."),

("Premières preuves : la structure porte de l'information", 90,
"Avant de construire TOPORET, j'ai réuni trois indices, présentés au chapitre 3. F1 : le gabarit de graphe a d'abord "
"été évalué hors de la médecine, sur l'orchestration de microservices et la détection d'anomalies, avec un F1 de 94,4 à "
"97,0 % dans l'étude de conférence. Mais la réévaluation sur dix exécutions montre honnêtement que les invariants "
"ajoutés apportent de l'interprétabilité et de la robustesse à la dérive, pas un gain de précision significatif. F2 : "
"sur APTOS 2019 en binaire, un modèle hybride CNN-GNN avec géométrie et topologie atteint 95,1 %, soit 7 points de plus "
"que le CNN, sur un seul découpage. F3 : la structure vasculaire globale est discriminante, ce que confirment trois "
"méthodes ; SAMF-GB, un modèle de boosting sur CPU, gagne 0,042 de QWK face à EfficientNet-B3 sur GPU. Ce n'est pas une "
"comparaison d'architectures à armes égales, mais cela justifie de tester un graphe de population qui intègre la "
"topologie."),

("Partie 4 – TOPORET", 15,
"Quatrième partie : TOPORET, qui porte les hypothèses H1 et H2."),

("TOPORET : un graphe de population à double similarité", 90,
"TOPORET relie les images dans un graphe de population. Deux images sont proches si elles se ressemblent pour le réseau "
"de convolution – c'est la similarité cosinus – et si leurs réseaux vasculaires ont une topologie proche – c'est le "
"terme exponentiel sur la distance entre leurs descripteurs topologiques. Le poids alpha vaut 0,6 et le seuil tau 0,65, "
"choisis par recherche sur grille avec les données de validation. La topologie joue un double rôle : dans les arêtes, "
"et dans les caractéristiques des nœuds, qui concatènent 1 536 dimensions visuelles et 6 dimensions topologiques. Je "
"précise le protocole, tel qu'il est dans le code publié : validation croisée en cinq plis, graine 42, au niveau de "
"l'image, et transductive, c'est-à-dire que les images de test sont présentes dans le graphe comme nœuds non étiquetés."),

("Décrire le réseau vasculaire par l'homologie persistante", 90,
"Concrètement, on améliore le contraste avec CLAHE, on extrait le squelette des vaisseaux, on prend ses points de "
"bifurcation comme nuage de points, puis on calcule une filtration de Vietoris-Rips avec la bibliothèque GUDHI, avec "
"une longueur d'arête maximale de 100 pixels. On résume ensuite les diagrammes par six nombres : pour les composantes H0 "
"comme pour les boucles H1, la persistance totale, l'entropie de persistance et le nombre de barres longues, une barre "
"étant longue si elle dépasse 5 % de la plus longue barre. Je précise honnêtement que la similarité compare ces six "
"statistiques, et non les diagrammes complets : c'est un résumé grossier de la topologie, moins coûteux, qui ne garde "
"que l'information de ces statistiques."),

("Résultats sur trois bases", 90,
"Sur les trois bases, TOPORET améliore toutes les métriques par rapport au même CNN : le QWK passe de 0,84 à 0,88 sur "
"Kaggle DR, de 0,83 à 0,90 sur Messidor-2, et de 0,82 à 0,86 sur APTOS 2019 ; la précision gagne de 1,5 à 2,3 points. "
"Le gain est le plus grand sur Messidor-2, dont l'acquisition standardisée donne des squelettes plus propres, et le plus "
"faible sur APTOS, dont la qualité d'image est plus variable. Une précaution importante : ces valeurs absolues sont "
"obtenues en transductif, avec des plis au niveau de l'image ; elles ne se comparent pas directement aux systèmes "
"publiés. Ce qui est valide, c'est la comparaison appariée avec le même CNN, sur les mêmes caractéristiques et les "
"mêmes plis."),

("H1 : des preuves convergentes mais limitées", 135,
"Le test de H1, fixé à l'avance, est un test t apparié sur les cinq plis, avec un seuil de 2,78 en valeur absolue. Sur "
"Kaggle DR, t vaut 3,82, avec p égal à 0,019 : le test est passé, et il l'est aussi sur Messidor-2 et APTOS. Mais cinq "
"plis, c'est peu : quatre degrés de liberté. J'ai donc ajouté deux analyses. Un bootstrap sur les différences donne un "
"intervalle de plus 0,021 à plus 0,059, qui exclut zéro. Un test de permutation exact donne p égal à un trente-deuxième, "
"le minimum possible, car le gain est positif dans chaque pli. Il reste un point : les plis partagent l'essentiel de "
"leurs données d'entraînement. J'ai appliqué la correction de Nadeau et Bengio. Après correction, aucune base seule ne "
"reste sous 0,05 : p vaut 0,070, 0,058 et 0,088. Les trois bases étant indépendantes, je les combine par la méthode de "
"Fisher, ce qui donne p égal à 0,015. Mon verdict est donc le suivant : des preuves convergentes mais limitées en faveur "
"de H1, dans les conditions évaluées, et non une preuve d'un avantage général des graphes de population."),

("D'où vient le gain : l'ablation", 75,
"L'ablation sur Kaggle DR isole chaque composant. Ajouter seulement les descripteurs topologiques aux nœuds, sans "
"graphe, donne plus 0,02 de QWK. Le graphe avec des caractéristiques visuelles seules, la topologie n'intervenant que "
"dans les arêtes, donne plus 0,03 : c'est le plus gros contributeur. Le modèle complet donne plus 0,04. Les "
"contributions sont donc complémentaires et à peu près additives. La matrice de confusion montre que l'amélioration la "
"plus nette se situe à la frontière entre les grades léger et modéré, la plus ambiguë aussi pour les experts humains. Je "
"précise que les tests de l'ablation ne sont pas corrigés du chevauchement des plis."),

("H2 : une association statistique faible mais robuste", 105,
"H2 est testée par une analyse de variance fixée à l'avance : la persistance totale H1 est comparée entre les cinq "
"grades, sur les 35 126 images de Kaggle DR. On obtient F égal à 84,3, avec p inférieur à 0,001, soit environ 23 fois la "
"valeur critique. Le seuil de décision, 0,005, est inférieur au niveau de Bonferroni pour six descripteurs. Et comme "
"les deux yeux d'un même patient ne sont pas indépendants, j'ai vérifié que même avec un échantillon effectif divisé par "
"deux, F resterait autour de 42. Mais la taille d'effet est petite : êta carré vaut environ 0,01, c'est-à-dire que le "
"grade explique environ 1 % de la variance. Ma conclusion est donc mesurée : c'est une association statistique robuste, "
"pas une causalité, et pas une signification biologique démontrée. Un lien avec la néovascularisation est plausible, "
"mais non vérifié ; j'emploie le terme de caractéristique topologique associée à la sévérité, et non de biomarqueur. "
"C'est cohérent avec l'ablation : la topologie est une information complémentaire."),

("TOPORET : les menaces à la validité", 90,
"J'expose moi-même les limites de TOPORET. Validité interne : le protocole est transductif ; les caractéristiques des "
"images de test, mais pas leurs grades, influencent leurs voisins pendant l'entraînement. Ce n'est donc pas équivalent à "
"une évaluation strictement inductive et indépendante du patient. Validité statistique : cinq plis, une graine, et une "
"significativité corrigée seulement pour les trois bases combinées. Validité externe : Kaggle DR et Messidor-2 "
"contiennent les deux yeux de chaque patient ; une image de test peut être reliée à l'œil adelphe du même patient. Deux "
"faits limitent ce risque : le gain est aussi observé sur APTOS, qui n'a qu'une image par patient, et DGTS montre un gain "
"du graphe en mode inductif. La réévaluation au niveau du patient est la première de mes perspectives."),

("Partie 5 – DGTS et DOTS", 15,
"Cinquième partie : les deux systèmes entraînés sur un corpus multi-source, et le test de H3."),

("Deux systèmes, un même corpus multi-source", 90,
"TOPORET laissait trois lacunes : un entraînement sur une seule source, une perte qui ignore l'ordre des grades, et une "
"sécurité non mesurée. DGTS et DOTS partagent le même corpus fusionné : APTOS 2019, 3 662 images, et KaggleDR-F1, 2 197 "
"images, soit 5 859 images, dont 309 sévères et 472 proliférantes. Pour ce corpus, le découpage par image est aussi un "
"découpage par patient. Un point d'honnêteté sur la provenance : KaggleDR-F1 contient exactement 60 % des effectifs "
"d'APTOS dans chaque grade. Un chevauchement entre les deux sources n'est donc pas exclu, et je lis la validation croisée "
"comme une preuve interne multi-source, peut-être optimiste. C'est pourquoi les conclusions reposent d'abord sur les "
"évaluations externes : EyePACS pour DGTS, 35 126 images, vérifié sans doublon, et Messidor-2 pour DOTS, 1 744 images "
"acquises en France."),

("DGTS : un graphe inductif régularisé par la topologie", 90,
"DGTS utilise un EfficientNet-B4 gelé, puis construit un graphe des huit plus proches voisins, par similarité cosinus, "
"uniquement sur les images d'entraînement : l'évaluation est donc inductive. Un GraphSAGE à deux couches traite ce "
"graphe. En parallèle, une branche d'homologie persistante décrit la forme du voisinage de chaque nœud dans l'espace "
"des caractéristiques. Un point souvent demandé : à l'inférence, ce descripteur vaut zéro ; la topologie agit comme un "
"régulariseur d'entraînement, à travers les gradients, et tous les chiffres sont mesurés dans ce réglage. Ce n'est pas "
"non plus la topologie des vaisseaux. L'ablation, sur APTOS, montre que le graphe apporte 0,018 de QWK, le régulariseur "
"0,030, les deux ensemble 0,052, et la fusion des sources 0,011 de plus. Un test de Wilcoxon de DGTS contre le CNN donne "
"p égal à 0,031."),

("DGTS sur données externes : la précision n'est pas la sécurité", 60,
"Appliqué sans réentraînement aux 35 126 images d'EyePACS, DGTS atteint un QWK de 0,925, soit une baisse de seulement "
"0,016 par rapport à la validation croisée. C'est une bonne généralisation. Mais son Sev-NR externe est d'environ 2,1 %, "
"légèrement au-dessus de l'objectif de 2 %. C'est l'une des leçons principales de la thèse : une bonne précision ne "
"garantit pas la sécurité, il faut mesurer les deux séparément. C'est ce qui motive DOTS. DGTS classe une image en 18,3 "
"millisecondes sur un GPU T4."),

("Les objectifs de H3, orientés dépistage", 60,
"Les objectifs de H3 ont été fixés à l'avance : un QWK d'au moins 0,80, un Sev-NR et un PDR-NR d'au plus 2 %, et aucune "
"confusion entre les grades 0 et 4. Ils viennent de l'article DOTS et s'inspirent de la pratique du dépistage : l'essai "
"pivot de l'IA autonome et le programme national anglais. Ce ne sont pas des seuils réglementaires. Et la définition est "
"explicite : une image sévère prédite « modérée » compte comme orientée, ce qui correspond à un parcours où le grade "
"modéré est déjà orientable ; une définition plus stricte donnerait des taux plus élevés."),

("DOTS : un classement ordinal orienté sécurité", 90,
"DOTS utilise un EfficientNet-B3 affiné de bout en bout, une projection, un encodeur de raffinement appelé FRE, une tête "
"ordinale CORAL et une augmentation au test sur cinq transformations. Une remarque honnête sur le FRE : il a la forme "
"d'un bloc transformer, mais il opère sur une séquence de longueur un ; l'attention n'y fait donc rien, et c'est en "
"pratique un perceptron résiduel normalisé. Je ne revendique aucun bénéfice de l'attention. L'ablation, à budget égal de "
"dix époques, montre la progression : 0,903 pour le CNN, 0,918 avec l'augmentation au test, 0,928 avec GraphSAGE, 0,937 "
"avec le FRE à la place du graphe, et 0,943 avec CORAL. Le graphe aide ici aussi, ce qui est cohérent avec H1, mais le "
"FRE fait mieux à moindre coût."),

("H3a : objectifs satisfaits en estimation ponctuelle", 90,
"Voici les résultats de DOTS. En interne, sur les 5 859 prédictions hors pli : un QWK de 0,953, un Sev-NR de 0,97 %, "
"soit 3 images sur 309, un PDR-NR de 1,27 %, soit 6 sur 472, et aucune confusion entre 0 et 4. Sur Messidor-2, jamais "
"vu pendant l'entraînement : un QWK de 0,931, un Sev-NR de 1,4 % et un PDR-NR de 1,9 %. Les performances observées "
"satisfont donc tous les objectifs, en estimation ponctuelle. Mais avec 309 images sévères, l'intervalle exact du Sev-NR "
"interne va de 0,2 à 2,8 % : il dépasse 2 %. Et les intervalles externes ne peuvent pas être calculés à partir des "
"effectifs publiés. La sécurité n'est donc pas démontrée. Au taux observé, il faudrait environ 650 images sévères pour "
"que la borne supérieure passe sous 2 %."),

("H3b : seule la chaîne complète atteint l'objectif", 90,
"H3b est testée sur Messidor-2. Chaque variante réduite échoue sur le Sev-NR : 5,8 % pour le CNN de base, 4,1 % avec "
"GraphSAGE, 3,7 % avec CORAL seul, 2,9 % avec l'augmentation au test seule. Seule la chaîne complète descend à 1,4 %. "
"Sur la figure, les barres représentent le QWK et les lignes les taux de non-orientation ; toutes les métriques évoluent "
"dans le même sens. L'enseignement est le suivant : le composant qui améliore le plus la sécurité n'est pas celui qui "
"améliore le plus le QWK. La limite : cette conclusion vaut pour les variantes de cette ablation, pas pour tout système "
"auquel il manquerait l'un de ces composants."),

("Comment DOTS se trompe : dans le sens de la sécurité", 60,
"La structure des erreurs compte autant que leur nombre. Sur 597 erreurs, 82,9 % sont entre grades adjacents, et aucune "
"ne traverse quatre grades. La confusion dominante est de « léger » vers « modéré » : une sur-orientation, pas une "
"sous-orientation. Les images sévères ont un rappel de 0,770 : DOTS prédit « sévère » plus souvent qu'il ne le faudrait, "
"ce qui est la bonne direction pour l'orientation. Après une mise à l'échelle de température, l'erreur de calibration "
"passe de 0,067 à 0,031, sans changer le QWK. Et le test de McNemar entre DOTS et le CNN donne p égal à 0,013."),

("Partie 6 – Ingénierie", 15,
"Sixième partie, brièvement : l'ingénierie logicielle."),

("Une conception attentive aux exigences réglementaires", 75,
"Le chapitre 6 relie l'architecture aux exigences de trois référentiels : IEC 62304 pour le cycle de vie du logiciel et "
"la vérification unitaire, ISO 14971 pour la gestion des risques, avec une AMDEC sur six dangers, et l'EU AI Act, qui "
"classe un tel système à haut risque. C'est un exercice d'ingénierie, pas une évaluation de conformité, ni un marquage "
"CE, ni une autorisation de la FDA. Le tableau d'état le montre clairement : les chaînes sont implémentées en code de "
"recherche et évaluées rétrospectivement ; le plan de vérification est spécifié ; l'adaptateur FHIR vers le dossier "
"patient est conçu mais pas implémenté ; il n'y a ni déploiement ni certification."),

("Partie 7 – Synthèse", 15,
"Dernière partie : la synthèse."),

("L'architecture des preuves", 60,
"Cette figure résume l'ensemble. En haut, l'objectif commun. Au milieu, les trois systèmes, qui apportent chacun un type "
"de preuve différent : une preuve transductive pour TOPORET, une preuve inductive et externe pour DGTS, une preuve "
"orientée dépistage pour DOTS. Toute cette preuve est rétrospective, obtenue sur des données publiques, en interne et "
"sur deux bases externes. Et en bas, en rouge, ce qui n'est pas encore évalué : un modèle unique à trois axes, TOPORET "
"au niveau du patient, plusieurs graines, une étude de lecteurs, des études prospectives et multi-sites, et la "
"validation clinique."),

("Ce qui est montré, soutenu, et non revendiqué", 90,
"Je distingue trois niveaux. Ce qui est montré directement, ce sont les quantités mesurées : le gain de TOPORET dans "
"chaque pli des trois bases, les gains du graphe et du régulariseur dans DGTS, le QWK externe de DGTS, les estimations "
"ponctuelles de DOTS qui satisfont les objectifs, et l'échec des quatre variantes réduites. Ce qui est soutenu sans être "
"établi de façon concluante : que l'information relationnelle améliore le classement, que la topologie porte une "
"information liée à la sévérité, et que la chaîne ordinale est assez sûre pour le dépistage. Enfin, ce que la thèse ne "
"revendique pas : la sécurité clinique, la conformité réglementaire, la prédiction longitudinale, ou une supériorité "
"générale des graphes. La contribution centrale est donc un ensemble structuré de preuves rétrospectives sur la façon "
"dont ces trois informations complètent les représentations visuelles profondes."),

("Limites et perspectives", 90,
"Les limites définissent directement les perspectives, par ordre de coût. D'abord, réévaluer TOPORET avec un découpage "
"par patient et un graphe construit à chaque pli : le code publié n'a besoin que d'un découpage groupé. Ensuite, répéter "
"les expériences sur plusieurs graines, et ajouter de nouvelles bases externes avec assez de cas sévères, environ 650. "
"Puis construire un modèle unifié – graphe, topologie vasculaire et tête CORAL – avec un mécanisme d'abstention fondé sur "
"les probabilités calibrées de DOTS. Viendraient ensuite une étude de lecteurs, une étude prospective pré-enregistrée, "
"dimensionnée sur environ 1 240 images sévères pour une puissance de 80 %, une vraie validation multi-site ou fédérée, "
"et enfin la validation clinique au sens du règlement européen sur les dispositifs médicaux."),

("Production scientifique", 30,
"Treize publications soutiennent la thèse : neuf sont acceptées ou publiées – sept articles de conférence, notamment à "
"ICAART, ICPRAM, ENASE, MIUA et IJCNN, et deux chapitres de livre chez Springer – et quatre articles de revue sont "
"soumis, dont un en révision à PLOS ONE et trois en cours d'évaluation."),

("Conclusion", 60,
"Pour conclure, sur les trois axes. R, les relations : des preuves en faveur de H1, cohérentes mais limitées par cinq "
"plis, et confirmées en mode inductif avec DGTS. S, la topologie : une association statistique faible mais robuste, et "
"un régulariseur utile dans DGTS, sans être un biomarqueur. O, l'ordinalité : DOTS satisfait les objectifs de dépistage "
"en estimation ponctuelle, en interne et sur Messidor-2, mais sa sécurité n'est pas encore démontrée. Les informations "
"relationnelles, topologiques et ordinales peuvent compléter les représentations visuelles profondes pour le classement "
"de la rétinopathie diabétique : voici la preuve, et voici exactement où elle s'arrête. Je vous remercie de votre "
"attention, et je suis prêt à répondre à vos questions."),
]

# (thème, question, réponse)
QA = [
("Vue d'ensemble", "Quelle est, en une phrase, la contribution de votre thèse ?",
"Un ensemble structuré de preuves rétrospectives sur la façon dont les informations relationnelles, topologiques et "
"ordinales peuvent compléter les représentations visuelles profondes pour le classement de la rétinopathie diabétique, "
"obtenu avec trois systèmes dédiés et des tests fixés à l'avance, et présenté avec ses limites."),
("Vue d'ensemble", "Pourquoi trois systèmes plutôt qu'un seul modèle ?",
"Parce que la thèse propose un cadre de recherche à trois axes. Chaque axe est porté par un système dédié pour que sa "
"contribution puisse être mesurée séparément, avec sa propre ablation. Réunir les trois dans un modèle unique, avec un "
"mécanisme d'abstention, est la prochaine étape ; ce n'est pas une affirmation de la thèse. Je ne dirai pas que DOTS "
"contient tout : dans sa version finale, il n'utilise ni le graphe ni la topologie."),
("Vue d'ensemble", "Les graphes de population existent déjà, par exemple chez Parisot et al. Qu'apportez-vous de nouveau ?",
"Les graphes de population existaient, et en rétinopathie ils reliaient les images par similarité visuelle seulement. "
"TOPORET ajoute des descripteurs de persistance du squelette vasculaire dans la similarité des arêtes et dans les "
"caractéristiques des nœuds ; à ma connaissance, cela n'avait pas été fait. DGTS ajoute un graphe inductif avec un "
"régulariseur topologique, et DOTS une évaluation orientée dépistage, où la sécurité est mesurée à côté du QWK et "
"vérifiée sur des données externes."),
("Vue d'ensemble", "Votre titre parle de prédiction de la sévérité. Prédisez-vous l'évolution de la maladie ?",
"Non. La tâche est transversale : prédire le grade ICDR à partir d'une seule image. Il n'y a pas de suivi du même patient "
"dans le temps. Un graphe de population temporel, reliant les visites successives d'un patient, fait partie des "
"perspectives."),
("H1 et TOPORET", "Votre test de H1 n'a que quatre degrés de liberté. Est-ce sérieux ?",
"C'est une limite, et je la traite comme telle. Le test fixé à l'avance donne t4 = 3,82 et p = 0,019. Comme cinq plis sont "
"peu, j'ai ajouté un bootstrap dont l'intervalle exclut zéro, et un test de permutation exact, p = 1/32, le minimum "
"possible. Comme les plis se chevauchent, j'ai appliqué la correction de Nadeau et Bengio : aucune base seule ne reste "
"sous 0,05, mais les trois bases indépendantes combinées par Fisher donnent p = 0,015. H1 est donc soutenue par des "
"preuves convergentes mais limitées, pas démontrée en général."),
("H1 et TOPORET", "TOPORET est transductif. N'y a-t-il pas une fuite d'information ?",
"Aucun grade des images de test n'est utilisé. Seules leurs caractéristiques influencent leurs voisins pendant "
"l'entraînement : c'est le cadre transductif standard, celui de Parisot et al. Ce n'est cependant pas équivalent à une "
"évaluation strictement inductive et indépendante du patient, et la thèse le dit. La preuve inductive de l'axe "
"relationnel vient de DGTS, dont le graphe est construit sur le seul pli d'entraînement."),
("H1 et TOPORET", "Kaggle DR et Messidor-2 contiennent les deux yeux. L'œil adelphe n'explique-t-il pas le gain ?",
"En partie, c'est possible : une image de test peut être reliée à l'autre œil du même patient, souvent de même grade. "
"Deux faits limitent cette crainte : le gain est aussi observé sur APTOS 2019, qui n'a qu'une image par patient, avec "
"+0,04 de QWK, et DGTS, évalué sur un corpus où le découpage est aussi par patient, montre un gain du graphe. La "
"réévaluation de TOPORET au niveau du patient est la première perspective."),
("H1 et TOPORET", "Pourquoi ne comparez-vous pas vos résultats absolus à ceux de la littérature ?",
"Parce qu'ils sont obtenus en transductif, avec des plis au niveau de l'image, ce qui diffère des jeux de test au niveau "
"patient ou externes de la plupart des études. J'ai donc retiré le classement face aux systèmes publiés. Ce qui est "
"valide, c'est la comparaison appariée avec le même CNN, sur les mêmes caractéristiques et les mêmes plis."),
("H1 et TOPORET", "Pourquoi α = 0,6 et τ = 0,65 ?",
"Ils ont été choisis par recherche sur grille, uniquement sur les données de validation. La thèse donne la grille de "
"sensibilité ; une version antérieure de conférence utilisait α = 0,7 et rapporte donc des chiffres légèrement "
"différents, ce que je signale pour qu'on ne mélange pas les deux configurations."),
("H1 et TOPORET", "Six statistiques résumées, est-ce vraiment de la topologie ?",
"C'est un résumé grossier des diagrammes de persistance, et je le dis. La similarité compare ces six statistiques et non "
"les diagrammes complets. Des représentations plus riches, comme les images de persistance ou des couches topologiques "
"apprenables, font partie des perspectives."),
("H2 et topologie", "η² ≈ 0,01 : l'effet n'est-il pas négligeable ?",
"Il est petit, et je le dis. L'association est robuste – F = 84,3 contre une valeur critique d'environ 3,72, et encore "
"environ 42 si l'échantillon effectif était divisé par deux –, mais le grade n'explique qu'environ 1 % de la variance. "
"La topologie seule n'explique pas la sévérité ; c'est une information complémentaire, cohérente avec son gain de 0,02 "
"de QWK dans l'ablation."),
("H2 et topologie", "La persistance H1 est-elle un biomarqueur de la néovascularisation ?",
"Non. Elle mesure les boucles du squelette vasculaire segmenté. Un lien avec la néovascularisation est plausible, mais "
"non démontré. J'utilise le terme de caractéristique topologique associée à la sévérité, et non celui de biomarqueur, "
"qui exigerait une validation clinique."),
("H2 et topologie", "Vous avez testé six descripteurs. Et les comparaisons multiples ?",
"Le seuil de décision a été fixé à 0,005, en dessous du niveau de Bonferroni 0,05/6, soit environ 0,008. La décision sur "
"la persistance H1 reste donc valide après correction."),
("H2 et topologie", "Pourquoi une ANOVA plutôt qu'une régression ordinale ?",
"H2 a été formulée comme une association entre une caractéristique définie à l'avance et cinq grades traités comme des "
"groupes ; l'ANOVA ne suppose rien sur l'écart entre les grades. Des modèles ordinaux ou de régression sont des analyses "
"complémentaires naturelles."),
("H2 et topologie", "La qualité de la segmentation ne biaise-t-elle pas la topologie ?",
"C'est une menace à la validité de construit, citée dans la thèse : un échec de la segmentation apparaîtrait comme un "
"changement de topologie. C'est d'ailleurs cohérent avec le gain plus faible sur APTOS, dont la qualité d'image plus "
"variable dégrade la squelettisation."),
("DGTS", "Que fait la branche topologique de DGTS à l'inférence ?",
"À l'inférence, le descripteur vaut zéro, donc sa projection aussi. La branche topologique n'agit qu'à travers les "
"gradients qui ont façonné la représentation du graphe pendant l'entraînement : c'est un régulariseur d'entraînement. "
"Tous les chiffres de DGTS sont mesurés dans ce réglage ; le gain est donc bien celui du modèle déployé."),
("DGTS", "La topologie de DGTS a-t-elle un lien avec les vaisseaux ?",
"Non. C'est la topologie des voisinages dans l'espace des caractéristiques ; ses boucles ne se lisent pas comme des "
"boucles vasculaires. L'interprétation anatomique de H2 ne concerne que TOPORET."),
("DGTS", "DGTS généralise bien mais rate l'objectif de sécurité. Qu'en retenez-vous ?",
"Que la précision et la sécurité doivent être mesurées séparément : un QWK de 0,925 sur EyePACS, mais un Sev-NR d'environ "
"2,1 %, légèrement au-dessus de 2 %. C'est ce qui a motivé la conception de DOTS."),
("DGTS", "L'ablation de DGTS est faite sur APTOS seul. Est-ce un deuxième corpus ?",
"Non, et je l'ai corrigé dans la thèse : APTOS fait aussi partie des bases de TOPORET. La différence avec TOPORET n'est "
"pas le corpus mais le protocole, inductif, et la forme de topologie. Le modèle final DGTS est entraîné sur le corpus "
"fusionné et évalué sur EyePACS."),
("DOTS et H3", "DOTS est-il sûr ?",
"Ses performances observées satisfont les objectifs de dépistage fixés à l'avance, en estimation ponctuelle, en interne "
"et sur Messidor-2. Mais la sécurité n'est pas démontrée : avec 309 images sévères, l'intervalle interne va de 0,2 à "
"2,8 %, au-dessus de 2 %, et les intervalles externes ne peuvent pas être calculés. Il faudrait environ 650 images "
"sévères au taux observé pour que la borne supérieure passe sous 2 %."),
("DOTS et H3", "D'où vient le seuil de 2 % ?",
"C'est un objectif d'évaluation de la thèse, repris de l'article DOTS et inspiré de la pratique du dépistage : l'essai "
"pivot de l'IA autonome, avec des critères fixés à l'avance, et les programmes nationaux qui suivent l'orientation des "
"formes menaçant la vision. Ce n'est pas un seuil réglementaire."),
("DOTS et H3", "Votre définition compte une image sévère prédite « modérée » comme orientée. N'est-ce pas trop permissif ?",
"Elle est moins stricte, et c'est dit avec la définition. Elle correspond à un parcours où le grade modéré est déjà "
"orientable. Une définition exigeant un grade prédit d'au moins 3 donnerait des taux plus élevés."),
("DOTS et H3", "Que signifie vraiment H3b ?",
"Sur Messidor-2, chaque variante réduite échoue sur le Sev-NR, de 5,8 à 2,9 %, et seule la chaîne complète atteint "
"1,4 %. La conclusion est limitée aux variantes de cette ablation : elle ne montre pas que tout système sans l'un de ces "
"composants échouerait."),
("DOTS et H3", "Votre FRE est présenté comme un bloc transformer. L'attention aide-t-elle ?",
"Non. Il opère sur une séquence de longueur un : les poids d'attention valent un, et il n'y a aucune interaction entre "
"éléments. Fonctionnellement, c'est un perceptron résiduel avec normalisation. Son bénéfice vient du goulot non linéaire "
"et du chemin résiduel ; je ne revendique aucun bénéfice de l'attention."),
("DOTS et H3", "CORAL n'apporte que 0,006 de QWK. Pourquoi le garder ?",
"Parce que le composant qui améliore le plus la sécurité n'est pas celui qui améliore le plus le QWK. CORAL donne le plus "
"grand gain de précision par point de QWK, concentre les erreurs sur les grades adjacents, et, avec les autres "
"composants, il est nécessaire pour atteindre l'objectif de Sev-NR sur données externes."),
("DOTS et H3", "Les tests de l'ablation de DOTS sont-ils rigoureux ?",
"Les tests t à un échantillon de l'ablation sont indulgents, car ils traitent la moyenne de référence comme fixe ; la "
"thèse le dit. La comparaison appariée de la chaîne complète avec le CNN est le test de McNemar, qui donne p = 0,013."),
("Données et protocole", "KaggleDR-F1 représente exactement 60 % d'APTOS dans chaque grade. Les sources sont-elles indépendantes ?",
"Ce n'est pas vérifié. Cette proportionnalité est compatible avec une dérivation de KaggleDR-F1 à partir d'APTOS ; la "
"validation croisée sur le corpus fusionné est donc lue comme une preuve interne multi-source, peut-être optimiste. Les "
"évaluations externes, EyePACS vérifié sans doublon et Messidor-2, ne sont pas concernées ; c'est sur elles que "
"reposent les conclusions."),
("Données et protocole", "Vos effectifs sont-ils des patients ou des images ?",
"Des images. Pour le corpus fusionné, le découpage est aussi un découpage par patient, car APTOS a une image par patient "
"et KaggleDR-F1 pas de paires bilatérales. Pour TOPORET sur Kaggle DR et Messidor-2, ce n'est pas le cas, et je le "
"signale."),
("Données et protocole", "Pourquoi une seule graine ?",
"Les plis ont été tirés avec la graine 42 pour la reproductibilité. La variabilité entre plis est rapportée partout, "
"mais la répétition sur plusieurs graines n'a pas été faite ; c'est la deuxième perspective."),
("Données et protocole", "Pourquoi EfficientNet et pas RETFound ou un Vision Transformer ?",
"La thèse isole la contribution des relations, de la topologie et de l'ordinalité sur un réseau fixe et très utilisé, "
"pour que les gains ne soient pas confondus avec un changement de réseau. Les modèles de fondation rétiniens comme "
"RETFound sont une perspective méthodologique."),
("Données et protocole", "Sans données démographiques, que dire de l'équité ?",
"Aucune des bases publiques ne fournit de métadonnées démographiques par image ; une analyse par sous-groupe n'a donc pas "
"été possible. C'est une limite déclarée, et une exigence de l'étude prospective prévue."),
("Chapitre 3", "Pourquoi évaluer le gabarit de graphe sur des microservices et la détection d'intrusions ?",
"Ces domaines ont des étiquettes sans ambiguïté et des graphes bien structurés, ce qui permettait de tester le gabarit "
"avant de l'appliquer aux patients. La réévaluation sur dix exécutions donne un résultat honnête : les invariants "
"apportent interprétabilité et robustesse à la dérive, pas un gain de précision significatif. Le transfert vers la "
"rétine est une analogie de conception, pas un résultat validé."),
("Chapitre 3", "Que montre SAMF-GB ?",
"Que des descripteurs structurels seuls contiennent déjà assez d'information pour rivaliser avec un réseau profond sur "
"cette tâche : +0,042 de QWK face à EfficientNet-B3, avec un classifieur sur CPU environ 13 fois moins coûteux. Ce n'est "
"pas une comparaison d'architectures à armes égales, puisque SAMF-GB utilise des caractéristiques précalculées."),
("Chapitre 6 et déploiement", "Votre système est-il conforme au règlement sur les dispositifs médicaux ou à l'EU AI Act ?",
"Je ne revendique aucune conformité. Le chapitre 6 est une mise en correspondance d'ingénierie orientée réglementation, "
"pas une évaluation de conformité, une évaluation clinique, un marquage CE ou une autorisation FDA. Le tableau d'état le "
"montre : l'adaptateur FHIR est conçu mais non implémenté, et il n'y a eu ni déploiement ni certification."),
("Chapitre 6 et déploiement", "À quoi ressemblerait l'étude suivante ?",
"Une étude de lecteurs prospective, multi-site et pré-enregistrée, dimensionnée sur les cas sévères : environ 1 240 "
"images sévères pour 80 % de puissance, soit environ 25 000 images dépistées à 5 % de prévalence des formes sévères, "
"avec un découpage par patient et un mécanisme d'abstention fondé sur les probabilités calibrées de DOTS."),
("Chapitre 6 et déploiement", "Votre système peut-il fonctionner en pratique ?",
"En laboratoire, oui : DGTS classe une image en 18,3 ms et DOTS en environ 75 ms sur un GPU T4. Mais sans adaptateur FHIR, "
"les résultats ne peuvent pas être écrits automatiquement dans le dossier patient ; c'est le principal écart entre les "
"chaînes de recherche et un produit clinique."),
("Questions pièges", "Alors, qu'avez-vous vraiment prouvé ?",
"Les expériences montrent directement les quantités mesurées, dans les protocoles décrits. Elles soutiennent, sans "
"l'établir de façon concluante, que l'information relationnelle améliore le classement, que la topologie porte une "
"information liée à la sévérité, et que la chaîne ordinale satisfait les objectifs de dépistage. Elles n'évaluent pas "
"la sécurité clinique, TOPORET au niveau du patient, la variabilité entre graines, l'équité, la prédiction "
"longitudinale ni la conformité réglementaire. C'est ce que résume le tableau GC.2."),
("Questions pièges", "Si vous aviez six mois de plus, que feriez-vous en premier ?",
"Je referais TOPORET avec un découpage par patient et un graphe construit à chaque pli, je répéterais toutes les "
"expériences sur plusieurs graines, et j'ajouterais une deuxième base externe avec assez de cas sévères. Ces trois "
"étapes répondent aux principales limites statistiques sans nouvelle infrastructure."),
("Questions pièges", "Vos articles publiés rapportent-ils les mêmes chiffres que la thèse ?",
"Les chiffres de TOPORET suivent le protocole de l'article PLOS ONE et de son code publié. La version courte à ENASE "
"utilisait une configuration antérieure, avec α = 0,7, et rapporte donc des valeurs différentes, par exemple F = 47,3 "
"pour l'ANOVA ; la thèse le signale explicitement pour que les deux ensembles de chiffres ne soient pas mélangés."),
("Questions pièges", "Quelle est la plus grande faiblesse de votre thèse ?",
"Le caractère transductif, au niveau de l'image, de l'évaluation de TOPORET, qui porte H1. Je l'ai atténuée par la "
"correction du chevauchement des plis, par le gain observé sur APTOS et par la preuve inductive de DGTS, et je propose "
"de la lever en premier, avec un découpage par patient."),
]
