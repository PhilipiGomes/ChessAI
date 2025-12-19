# ChessAI

ChessAI é um projeto em Python que implementa uma **rede neural para avaliação de posições de xadrez**.  
O objetivo é treinar um modelo capaz de aproximar avaliações de engines como o **Stockfish**, usando posições de xadrez convertidas em vetores numéricos.

O projeto foi pensado para ser simples, extensível e experimental, permitindo testar arquiteturas, funções de perda e diferentes fontes de dados.

---

## Objetivo

Dado uma posição de xadrez, o modelo aprende a prever uma avaliação numérica (por exemplo, em centipawns ou pawns), podendo ser usado para:

- Avaliação estática de posições
- Análise de partidas
- Base para um motor de xadrez híbrido
- Experimentos com aprendizado de máquina em xadrez

---

## Funcionalidades

- Treinamento de rede neural feedforward
- Suporte a múltiplas camadas ocultas configuráveis via CLI
- Treinamento com mini-batch SGD
- Estrutura preparada para loss customizada (ex.: Huber)
- Código simples e totalmente em NumPy
- Fácil adaptação para treino online ou distribuído

---

## Estrutura do Projeto

```

ChessAI/
├── app.py              # Script principal (treino / execução)
├── src/                # Código-fonte
│   ├── model.py        # Definição da rede neural
│   ├── train.py        # Loop de treino
│   ├── data.py         # Pré-processamento / encoding
│   └── utils.py        # Funções auxiliares
├── tests/              # Testes básicos
├── requirements.txt    # Dependências
└── README.md           # Documentação

````

---

## Instalação

Recomenda-se o uso de um ambiente virtual.

```bash
git clone https://github.com/PhilipiGomes/ChessAI.git
cd ChessAI
pip install -r requirements.txt
````

---

## Uso Básico

### Treinamento

Exemplo de treino com duas camadas ocultas:

```bash
python app.py \
  --data path/para/dataset.npy \
  --hidden 512 256 \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 50
```

### Camadas ocultas

O argumento `--hidden` define a arquitetura da rede:

* `--hidden` → nenhuma camada oculta (modelo linear)
* `--hidden 256` → uma camada oculta
* `--hidden 512 256 128` → três camadas ocultas

---

## Dados de Entrada

Cada posição deve ser convertida em um **vetor numérico fixo** antes do treino.
O projeto não impõe um encoding específico, mas exemplos comuns incluem:

* Bitboards
* Planos por peça
* Representações binárias da posição
* Features manuais (material, mobilidade, etc.)

A consistência do encoding entre treino e inferência é obrigatória.

---

## Rótulos (Targets)

Os rótulos normalmente são avaliações de engine:

* Centipawns (`+34`, `-120`, etc.)
* Avaliações de mate devem ser tratadas separadamente

---

## Função de Perda

Embora MSE funcione, o recomendado para esse problema é:

* **Huber (Smooth L1)**: mais robusta a outliers
* Melhor comportamento com avaliações extremas e ruído do Stockfish

A implementação permite substituir a loss facilmente no loop de treino.

---

## Avaliação do Modelo

Métricas recomendadas:

* MAE / RMSE
* Correlação de ranking (Spearman)
* Acurácia em comparações par-a-par
* Curvas de loss por época

---

## Testes

Os testes básicos ficam em `tests/`.

```bash
pytest
```

---

## Autor

Philipi Gomes
Projeto experimental e educacional focado em IA aplicada ao xadrez.