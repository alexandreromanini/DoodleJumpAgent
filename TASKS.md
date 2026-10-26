# Tâches

- [x] Phase 1 — Analyse du jeu ([docs/01-analyse-du-jeu.md](docs/01-analyse-du-jeu.md))
- [x] Phase 2 — Proposition d'approches ([docs/02-approches.md](docs/02-approches.md)), validée : planificateur puis PPO
- [x] Phase 3 — Implémentation
  - [x] Jeu en sous-module, API `reset_game`/`step`, ressort sorti de `draw()`, index de quadrant remis à zéro
  - [x] Test d'équivalence avec le jeu original (même seed, mêmes entrées, rendu contre headless)
  - [x] Environnement Gymnasium headless, tests pytest
  - [x] Planificateur à modèle exact, testé contre le jeu
  - [x] Scripts `train`, `evaluate`, `play`, `record`, `plot`
- [x] Phase 4 — Évaluation et itérations
  - [x] Gel du jeu (tirage des tuiles sans fin) détecté et géré
  - [x] Planificateur v1 → v5 (36k → 96k en validation), essais sans gain documentés
  - [x] PPO depuis zéro (20 M pas)
  - [x] PPO initialisé par imitation du planificateur (DAgger) : pas de gain, documenté
  - [x] Évaluation finale sur 100 seeds d'évaluation : aléatoire, planificateur, PPO
  - [x] Courbes et analyse des fins de partie
- [ ] Phase 5 — README, GIF, licence : faits. Reste la publication (fork du jeu, URL du sous-module, dépôt GitHub)
