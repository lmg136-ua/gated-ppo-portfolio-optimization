# Demo visual del modelo

Mini web estatica para explicar la arquitectura `PPO_gated_full` sin lanzar entrenamientos.

## Uso

1. Regenera el payload si cambian resultados o configuracion:

   ```powershell
   python scripts/build_demo_data.py
   ```

2. Abre `demo/index.html` en el navegador o sirve la carpeta:

   ```powershell
   python -m http.server 8765 -d demo
   ```

   Despues entra en `http://localhost:8765`.

La demo usa `config/config.yaml`, `dataset/raw/*` y `results/final_v2/*`. No reconstruye pesos diarios porque esos artefactos no estan versionados en el repositorio.
