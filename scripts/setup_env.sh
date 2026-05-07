#!/bin/bash
# ============================================================
# setup_env.sh — Instalación de dependencias
# ============================================================
# Uso: bash scripts/setup_env.sh

echo "=== TFG RL Portfolio — Setup ==="

# Verificar Python 3.10+
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $python_version"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt --quiet

echo ""
echo "=== Verificando instalaciones ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
python3 -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python3 -c "import cvxpy; print(f'CVXPY: {cvxpy.__version__}')"
python3 -c "import yfinance; print(f'yfinance: {yfinance.__version__}')"

echo ""
echo "=== Setup completado ==="
echo "Activar entorno: source venv/bin/activate"
echo "Primer test:     python experiments/run_experiment.py --mode demo"
