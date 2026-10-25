# Phase 2 — Approches envisagées

Ce qui rend ce jeu particulier, d'après la phase 1 : l'état est **entièrement observable**, la physique est
**simple et déterministe** (en dehors des tirages aléatoires de génération), et la simulation tourne vite
(environ 15 000 frames/s par cœur). Une politique naïve atteint déjà une médiane d'environ 8 000. La difficulté
vient de la montée en score : tuiles pièges, ennemis qui arrivent par le haut et raréfaction des tuiles.

## A. Planificateur à modèle exact (baseline)

À chaque frame, l'agent fait du *model predictive control* :

1. **Modèle** : il réimplémente en quelques lignes la physique du joueur en coordonnées écran, y compris la
   troncature `int()` de la caméra et l'ordre des collisions. Les tuiles mobiles sont extrapolées (mouvement
   linéaire avec rebonds). Un test unitaire vérifie que les prédictions coïncident frame par frame avec le jeu.
2. **Atteignabilité** : comme le mouvement horizontal (5 px/frame) est indépendant du vertical, une tuile est
   atteignable si la distance horizontale, avec wrap, est inférieure ou égale à 5 × le nombre de frames avant que
   la trajectoire ne la croise en descendant.
3. **Recherche** : il explore en profondeur 2 ou 3 (tuile cible, puis tuiles atteignables depuis celle-ci).
   L'évaluation récompense le gain de hauteur. Elle pénalise ou interdit les tuiles cassées, les tuiles
   explosives (à cause de la fuite de tuiles), les trajectoires qui croisent un ennemi en montant et les trous
   noirs. Elle favorise les ressorts, les trampolines et les jetpacks.
4. **Tir** : il tire dès qu'un monstre ou un OVNI est au-dessus de lui dans l'axe.

Le modèle est réimplémenté plutôt qu'obtenu en clonant l'état du jeu (`deepcopy`), parce que les objets pygame
(Surface, Sound) ne se copient pas proprement, et parce qu'un modèle explicite est plus rapide et plus lisible.

## B. PPO (Stable-Baselines3) sur un environnement Gymnasium

- **Observation** (vecteur normalisé d'environ 200 dimensions) :
  - Joueur : `vy`, x encodé en sin/cos (à cause du wrap), état des bonus (bouclier, chaussures, vol).
  - Les **N = 12 tuiles les plus proches**, au-dessus **et en dessous** : dx avec wrap, dy, type en one-hot,
    vitesse, bonus en one-hot, plus un masque de présence.
  - Les **K = 3 ennemis** : dx, dy, type, vitesse.
- **Actions** : `Discrete(6)`, soit {gauche, rien, droite} × {tir, pas de tir}, avec une répétition d'action sur
  4 frames pour réduire l'horizon.
- **Récompense** : `Δscore / 100` à chaque pas, qui correspond exactement à l'objectif (camper rapporte 0), et
  `−1` à la mort. Pas de bonus de proximité des tuiles, pour éviter le *reward hacking*.
- **Fin d'épisode** : à la mort. Troncature au bout de N frames, ou après environ 20 s sans gain de hauteur
  (contre le camping).
- **Risque principal** : les ennemis et les tuiles spéciales n'apparaissent qu'à haut score, donc l'agent les
  rencontre peu en début d'entraînement.

## C. Neuroévolution (NEAT ou CMA-ES sur un petit MLP)

Elle utiliserait les mêmes observations et actions, avec comme fitness le score moyen sur plusieurs graines.
C'est simple à paralléliser et ça fonctionne bien sur ce type de jeu. En revanche la fitness est bruitée (forte
variance entre graines) et l'approche est très répandue dans les projets hobby (« NEAT joue à Flappy Bird »).

## Comparatif

| Critère | A. Planificateur | B. PPO | C. Neuroévolution |
|---|---|---|---|
| Score attendu | **Le plus haut** (plusieurs dizaines de milliers ?), limité par les ennemis et la fuite de tuiles | Moyen à haut, probablement sous A | Moyen |
| Temps de dev | 2–3 j (modèle + tests + recherche) | 2–3 j (env + obs + tuning) | 1–2 j de plus que B |
| Entraînement | Aucun | ≈ 1–3 h pour 20–50 M pas (12 cœurs, à mesurer) | ≈ 1–2 h |
| Robustesse | Bonne si le modèle est exact. Elle casse sur les cas non modélisés | Généralise sans modèle, mais elle est sensible aux réglages et au *seed* | Variance élevée |
| Intérêt portfolio | Montre la compréhension du système (modèle, tests, recherche) | La compétence la plus recherchée (RL, Gymnasium, SB3) | Faible, déjà vu |

## Recommandation : A puis B, comparés à un agent aléatoire

Je suis votre préférence de départ, et C est écarté.

1. **A en premier** : c'est la référence solide. C'est aussi lui qui sert à valider l'environnement, puisqu'un
   agent qui joue bien prouve que l'API d'état et d'actions est juste.
2. **B ensuite**, sur le même environnement, avec les mêmes graines d'évaluation (100 parties, graines distinctes
   de l'entraînement).
3. **L'histoire à raconter** : dans un jeu entièrement observable et déterministe, un planificateur qui connaît
   la physique est très difficile à battre. La question devient : jusqu'où un agent PPO, **qui ne connaît pas les
   règles**, s'en approche-t-il, et où échoue-t-il ? Le comparatif est honnête dans les deux sens.
4. **Si PPO plafonne nettement**, piste bonus : initialiser PPO par imitation du planificateur (*behavior
   cloning*), puis l'affiner en RL.

## Décisions à prendre avant la phase 3

1. **Licence et provenance du jeu.** Le dépôt LunaTMT n'a **pas de licence** (tous droits réservés par défaut),
   et ses sprites et sons viennent du Doodle Jump commercial (Lima Sky). Recopier le code et les assets dans un
   dépôt public est risqué. Je propose de **forker LunaTMT/Doodle-Jump sur GitHub**, d'y appliquer les
   modifications autorisées (a), (b) et (c), et de l'intégrer en **sous-module git**. Le fork reste sous la
   paternité de l'auteur et n'est pas redistribué hors de GitHub.
2. **Python 3.12**, avec pygame 2.6.1, SB3 et torch figés. Il est disponible sur la machine, contrairement à
   pygame pour 3.14.
3. **Limite d'évaluation** : un bon agent peut ne jamais mourir. Je propose un plafond de **100 000 frames par
   partie** (environ 28 min de jeu réel), indiqué à côté des résultats.
