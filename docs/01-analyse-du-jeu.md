# Phase 1 — Analyse du jeu

Jeu analysé : [LunaTMT/Doodle-Jump](https://github.com/LunaTMT/Doodle-Jump) (commit `3740388`), Python + pygame,
environ 3 100 lignes. Toutes les valeurs ci-dessous viennent de la lecture du code. Celles marquées **(mesuré)**
ont été vérifiées en exécutant le jeu original, sans aucune modification, avec des entrées scriptées.

## 1. Architecture et boucle de jeu

| Fichier | Rôle |
|---|---|
| `main.py` | `Game().run()` |
| `game.py` | Classe `Game` : états (menu / options / jeu / fin), génération des tuiles et des ennemis, boucle principale |
| `Sprites/player.py` | Joueur : entrées, physique, score, caméra, mort par chute ; `Bullet` |
| `Sprites/tile.py` | 7 types de plateformes et le tirage des bonus |
| `Sprites/monster.py`, `blackhole.py`, `ufo.py` | Ennemis |
| `Sprites/Power_ups/*.py` | Jetpack, hélice, bouclier, ressort, trampoline, chaussures à ressort |
| `Buttons/*.py` | Menus, pause, rejouer (souris uniquement) |
| `texture.py`, `Assets/` | Pack de textures (11 thèmes), sons, images |

La boucle (`Game.run`) enchaîne `handle_events()`, `update()`, `draw()`, `frame += 1` et `clock.tick(60)`.

- **Pas de temps fixe, sans delta time.** Toute la physique est exprimée en px/frame. `clock.tick(60)` ne fait que
  limiter la vitesse d'exécution : le retirer ne change pas la simulation.
- Dans l'état « jeu », `update()` appelle dans cet ordre : génération des tuiles, génération des ennemis, balles,
  **plateformes (collisions tuile-joueur et bonus-joueur)**, joueur (mouvement, gravité, caméra), puis monstres,
  trous noirs et OVNI (collisions ennemi-joueur). Les collisions avec les tuiles utilisent donc les positions de
  la frame précédente.

## 2. Physique

| Grandeur | Valeur |
|---|---|
| Gravité | `+0.4` px/frame² ; intégration `v += g` puis `y += v` (Euler semi-implicite) ; pas de vitesse terminale |
| Saut normal | `v = -15` donne une apogée de **≈ 274 px** en 37 frames |
| Ressort / chaussures à ressort | `-23` donne ≈ 650 px (chaussures : 5 sauts) |
| Trampoline | `-30` donne ≈ 1 110 px |
| Hélice / jetpack | `-55` / `-65` donne ≈ 3 750 / 5 250 px, invulnérable tant qu'il monte |
| Horizontal | **5 px/frame**, sans inertie ; gauche et droite ensemble s'annulent |
| Bords | **Wrap-around** sur le centre du joueur (`x > 640` donne `x = 0`, et inversement). Les hitboxes, elles, ne se replient pas |
| Écran | 640 × 900. Joueur 62 × 60, tuiles 57–60 × 15–17, tuile déplaçable 80 × 35 |

**Caméra.** `player.y` est une coordonnée monde : il vaut −900 au départ et diminue quand on monte. Tant que
`player.y < 390`, le rect du joueur reste figé à l'écran (`rect.y = 390`) et c'est le décor qui défile, **vers le
haut comme vers le bas** : la caméra suit aussi la chute. Le décalage appliqué au décor est tronqué par `int()` à
chaque frame, ce qui crée un léger glissement entre la position monde et l'écran. Un modèle qui veut prédire
exactement doit reproduire cette arithmétique.

**Atterrissage.** Il y a rebond dès que le rect du joueur **chevauche** une tuile pendant la descente
(`velocity_y > 0.4`). Il n'est pas nécessaire d'arriver par le dessus : toucher une tuile avec le corps en
descendant suffit. La hitbox est donc généreuse.

## 3. Entités et génération

### Plateformes

Le jeu en maintient 26 au total (`while Tile.total <= 25`). Elles apparaissent au-dessus de l'écran
(`y ∈ [−920, 20]`) en alternant 4 quadrants, par tirage avec rejet : pas de chevauchement et au moins 120 px
entre centres. Le tirage n'a pas de limite d'essais.

| Type | Poids | Règle |
|---|---|---|
| `Tile` (normale) | 9 999 999 → 250 → 50 → 25 | Rebond |
| `MovingTile` | 5 | Va-et-vient horizontal, 1–4 px/frame, bornes aléatoires (≥ 150 px). Jamais de bonus |
| `ShiftingTile` | 5 | Après un rebond, glisse à 5 px/frame vers un x aléatoire. Même sprite que l'explosive |
| `MoveableTile` | 1 | Peut être glissée une fois à la souris par le joueur |
| `DisappearingTile` | 5 | Un seul rebond, puis disparaît |
| `BrokenTile` | 5 | **Aucun rebond** (piège) : elle casse et tombe |
| `ExplodingTile` | 5 | Le moindre contact, même par le dessous, la fait exploser 35 frames plus tard. Ses bonus sont inertes |

Le poids de la tuile normale baisse avec le score, mis à jour à chaque rebond. Les tuiles spéciales représentent
environ **0 %** des tirages jusqu'à 2 500, **9 %** de 2 500 à 10 000, **34 %** de 10 000 à 25 000 et **51 %**
au-delà. La branche 5 000–10 000 du code (`5000 < score <= 1000`) n'est jamais vraie, et la branche > 50 000 est
inatteignable.

### Bonus

Ils sont tirés à la création d'une tuile, sauf pour les tuiles cassées et mobiles : jetpack 0,1 %, trampoline 2,1 %,
ressort 7,3 %, hélice 0,8 %, bouclier 5,2 % (absorbe un choc), chaussures 1,0 %, rien 83,4 %.

### Ennemis

Au plus 1 ennemi jusqu'à un score de 10 000, puis 2, puis 3 au-delà de 25 000. À chaque frame, la probabilité d'en
faire apparaître un vaut `3w / (3w + 100)` avec `w = score / 100 000` : environ 0,03 %/frame à 1 000 et 0,3 %/frame
à 10 000. Le type est équiprobable et l'ennemi apparaît juste au-dessus de l'écran.

| Ennemi | Mouvement | Contact |
|---|---|---|
| Monstre (5 sprites) | Rebond, haut-bas ou côté à côté, 1–5 px/frame | En descendant : il meurt et on rebondit. En montant : **mort** |
| Trou noir | Immobile | **Mort** quelle que soit la direction |
| OVNI | Trajectoire en 8 | En descendant : il meurt. En montant : **mort** (aspiré) |

Le jetpack et l'hélice rendent invulnérable. Les balles tuent les monstres et les OVNI, pas les trous noirs.

## 4. Score et défaite

- **Score** = `max(|player.y| − 900)`, soit la hauteur maximale atteinte en pixels depuis le départ (un flottant,
  affiché en entier). Tuer un ennemi ne rapporte rien.
- **Défaite** : (1) chute, quand `rect.top ≥ 900`. Comme la caméra suit la chute, la mort n'est constatée
  qu'**environ 100 frames après** le dernier rebond raté. (2) Monstre ou OVNI touché en montant. (3) Trou noir.
  Dans les trois cas `player.dead = True`, puis l'état passe à `end_game` et le score ne bouge plus.

## 5. Actions du joueur

| Action | Source dans le code |
|---|---|
| Gauche / droite | `pygame.key.get_pressed()` (`K_LEFT`/`K_RIGHT`), lu dans `Player.update_movement` |
| Tir | Événement `KEYDOWN` (Espace / Haut) ou clic, une balle par appui, impossible en vol |
| Déplacer une `MoveableTile` | Glisser-déposer à la souris, une fois par tuile |
| Pause | Bouton souris |

Proposition d'espace d'actions : {gauche, rien, droite} × {tir, pas de tir}, soit 6 actions. Le glisser-déposer
et la pause en sont exclus.

## 6. Points d'accroche

- **État** : tout est dans des objets Python lisibles directement. `game.player` (`x`, `y`, `velocity_y`,
  `falling`, `dead`, `score`, `using_*`), `game.platforms` (rect, type, `power_up`, vitesse, cible de glissement),
  `game.monsters`, `game.blackholes`, `game.UFOs` (rect, vitesse, direction) et `game.bullets`.
- **Actions** : il suffit de remplacer la lecture de `pygame.key.get_pressed()` et d'appeler `player.shoot()`.
  C'est la modification (b).
- **Reset** : on contourne les menus en reproduisant ce que fait `PlayButton` :
  `initialise_game_weights()` puis `initialise_game_objects()`.
- **Pas de simulation** : `update()` suivi de `frame += 1`. Ce compteur sert au jeu, il est remis à 0 à chaque rebond.

## 7. Ce qui complique l'entraînement

1. **Aléatoire** : tout passe par le module global `random`, sans graine. `random.seed(s)` suffit à rendre une
   partie reproductible **(mesuré : traces identiques au bit près sur des parties rejouées)**. En revanche l'état
   est global, et pourrait être consommé par une autre bibliothèque du même processus.
2. **État global** : l'écran est créé, les images chargées et le mixer initialisé **à l'import**. Il y a des
   compteurs de classe (`Tile.total`, `Player.high_score`) et les chemins sont relatifs au répertoire courant.
   Conséquence : **une seule partie par processus**, donc des environnements vectorisés en sous-processus. Le mode
   sans fenêtre fonctionne avec `SDL_VIDEODRIVER=dummy` et `SDL_AUDIODRIVER=dummy`.
3. **Logique dans `draw()`** : le ressort passe à l'état « utilisé » et décale sa hitbox de 20 px dans
   `Spring.draw`, et la transition vers l'écran de fin a lieu dans `Game.draw`. Le mode headless doit garder ces
   effets. Sur 10 graines × 5 000 frames, les traces avec et sans rendu sont identiques **(mesuré)**, mais le cas du
   ressort réutilisé y est rare. Le test d'équivalence devra le couvrir explicitement.
4. **Temps réel** : il n'y a aucune dépendance, et la vitesse est bonne : **≈ 570 frames/s avec rendu contre
   13 000 à 16 000 sans** sur un cœur **(mesuré)**.
5. **Bugs qui pèsent sur la difficulté** (on n'y touche pas, car ce sont des règles du jeu) :
   - Une tuile explosive détruite ne décrémente pas `Tile.total`, si bien que le nombre de tuiles réelles diminue
     au fil de la partie (mesuré : 26 comptées pour 24 réelles).
   - Le tirage avec rejet peut boucler à l'infini si un quadrant est saturé.
   - Le bonus chaussures, une fois ramassé, reste actif mais invisible au même endroit.
6. **Épisodes très longs** : une politique naïve qui vise la tuile la plus proche en dessous atteint déjà
   **760 à 17 670** (10 graines), et peut aussi « camper » sur une tuile sans jamais mourir. Il faudra une limite
   de frames et un arrêt en l'absence de progrès.
7. **Python** : pygame 2.6.1 n'a pas de wheel pour Python 3.14 (le Python système), mais il s'installe sur
   Python 3.12, qui est disponible.
