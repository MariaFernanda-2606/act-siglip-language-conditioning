# ACT SigLIP Annotation Pipeline

Este repositório prepara anotações temporais e linguísticas para demonstrações robóticas da tarefa de pegar um cubo colorido e colocá-lo na gaveta da cor correspondente. O resultado final é um **manifesto linguístico**: um arquivo que associa cada intervalo de uma demonstração a uma subtarefa em linguagem natural. Na próxima etapa, esse texto será convertido em embeddings SigLIP para condicionar o ACT.

As três tarefas usam cubos e gavetas `red`, `green` ou `yellow`. Cada demonstração é organizada nas cinco fases canônicas:

```text
pick → move → open → place → close
```

O método final é o **híbrido em vocabulário aberto**:

```text
Qwen2.5 VL 7B Instruct ──> frases livres sobre as ações
                                  │
                                  ├── alinhamento semântico por SigLIP
                                  │
cinemática do robô ───────────> fronteiras temporais confiáveis
                                  │
                                  ▼
                    manifesto linguístico para o ACT
```

O Qwen fornece linguagem natural quando consegue descrever uma ação. A cinemática define os limites temporais, pois foi mais precisa para detectar transições. Quando o Qwen omite uma fase, um fallback baseado na fase cinemática e nos metadados da tarefa completa a anotação.

## Estrutura do repositório

```text
.
├── configs/evaluation_split.json                # 15 episódios de desenvolvimento, 30 de teste
├── datasets_annotation/
│   ├── evaluation_ground_truth/                 # referência manual
│   └── language_manifests/open_vocab/           # manifesto final para o ACT
├── prompts/cube_drawer_subtasks.txt             # prompt do Qwen fechado
├── scripts/
│   ├── run_qwen_annotation.sh                   # Qwen fechado
│   ├── run_qwen_open_vocab.sh                   # Qwen aberto
│   ├── segment_kinematic_subtasks.py            # segmentação cinemática
│   ├── rebind_open_vocab_semantics.py           # híbrido aberto
│   ├── build_open_vocab_language_manifest.py    # manifesto JSONL
│   ├── evaluate_subtasks.py                      # métricas temporais
│   ├── evaluate_open_vocab_semantics.py         # métricas semânticas
│   ├── plot_method_comparison.py                 # figuras de comparação
│   └── plot_all_method_timelines.py              # timelines qualitativas
└── outputs/                                      # resultados gerados localmente
```


## Configuração reproduzível

Após clonar o repositório, defina caminhos locais genéricos. 

```bash
git clone <URL_DO_REPOSITORIO> act-siglip
cd act-siglip
conda activate lerobot_annotation

export PROJECT_ROOT="$(pwd)"
export DATA_ROOT="/caminho/para/dados"
export DATASET_ROOT="$DATA_ROOT/place_the_cubes_in_the_drawers_v3"
export GROUND_TRUTH="$PROJECT_ROOT/datasets_annotation/evaluation_ground_truth/manual_ground_truth_45_episodes_canonical.json"
export SPLIT="$PROJECT_ROOT/configs/evaluation_split.json"

export QWEN_API_BASE="http://127.0.0.1:8000/v1"
export QWEN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export CAMERA_KEY="observation.images.cam_low"
```

Valide dataset e servidor:

```bash
python -c "import json; print(json.load(open('$DATASET_ROOT/meta/info.json'))['codebase_version'])"
curl -s "$QWEN_API_BASE/models"
```

O dataset deve reportar `v3.0`. Os experimentos deste projeto usaram `Qwen/Qwen2.5-VL-7B-Instruct`. Para os scripts com SigLIP, instale as dependências necessárias no ambiente:

```bash
pip install sentencepiece protobuf transformers torch matplotlib
```

Na primeira execução, `google/siglip-base-patch16-224` será baixado para o cache do Hugging Face.

## Dados e protocolo de avaliação

O ground truth manual contém 45 episódios, todos divididos em `pick`, `move`, `open`, `place` e `close`.

| Subconjunto | Episódios | Uso |
|---|---:|---|
| Desenvolvimento | 15 | Ajustar parâmetros, inspecionar erros e depurar |
| Teste | 30 | Métricas finais; não deve orientar novos ajustes |

A divisão está em [`configs/evaluation_split.json`](configs/evaluation_split.json). Essa separação evita escolher parâmetros de suavização, duração mínima ou alinhamento nos mesmos episódios usados para declarar o resultado final.

Os episódios `182`, `194` e `230` possuem apenas um frame. Como velocidade requer ao menos dois instantes, eles não podem ser segmentados cinematicamente. Portanto, o conjunto completo utilizável possui 261 episódios.

## Pipeline por etapa

### 1. Diagnóstico dos sinais cinemáticos

O script `plot_kinematic_boundaries.py` visualiza movimentos do braço e garra com fronteiras manuais. Use um episódio de desenvolvimento antes de ajustar a segmentação.

```bash
python scripts/plot_kinematic_boundaries.py \
  --root "$DATASET_ROOT" \
  --ground-truth "$GROUND_TRUTH" \
  --episode 0 \
  --output outputs/diagnostics/kinematics/episode_000.png
```

### 2. Baseline Qwen fechado

`run_qwen_annotation.sh` chama o LeRobot Annotate com o prompt `prompts/cube_drawer_subtasks.txt`. O modelo deve emitir rótulos fixos com fase e cor, como `pick_red_cube` e `place_red_cube_in_red_drawer`.

```bash
export QWEN_CLOSED_ROOT="$DATA_ROOT/qwen_closed_test"

./scripts/run_qwen_annotation.sh \
  --dataset-root "$DATASET_ROOT" \
  --out-root "$QWEN_CLOSED_ROOT" \
  --model-id "$QWEN_MODEL" \
  --api-base "$QWEN_API_BASE" \
  --camera-key "$CAMERA_KEY" \
  --episodes '[5,7,11,16,21,35,40,49,55,67,81,84,87,89,91,94,96,100,101,249,107,113,150,155,161,163,166,167,169,171]'

python scripts/export_lerobot_qwen_predictions.py \
  --root "$QWEN_CLOSED_ROOT" \
  --output outputs/qwen/qwen7b_test_predictions.json

python scripts/evaluate_subtasks.py \
  --ground-truth "$GROUND_TRUTH" \
  --predictions outputs/qwen/qwen7b_test_predictions.json \
  --output outputs/qwen/qwen7b_test_metrics.json
```

Esse método é somente baseline: o Qwen tenta resolver simultaneamente linguagem, classes e limites temporais.

### 3. Segmentação cinemática

`segment_kinematic_subtasks.py` é o segmentador temporal. Ele usa estados, velocidades e sinais da garra, aplica suavização e procura cinco segmentos ordenados. Os protótipos são obtidos a partir dos episódios de desenvolvimento.

Parâmetros selecionados:

```text
smoothing-s = 1.0 s
min-duration-s = 2.5 s
```

Linha de base cinemática no teste:

```bash
python scripts/segment_kinematic_subtasks.py \
  --root "$DATASET_ROOT" \
  --ground-truth "$GROUND_TRUTH" \
  --split "$SPLIT" \
  --subset test \
  --smoothing-s 1.0 \
  --min-duration-s 2.5 \
  --output outputs/kinematic/test/kinematic_test_predictions.json

python scripts/evaluate_subtasks.py \
  --ground-truth "$GROUND_TRUTH" \
  --predictions outputs/kinematic/test/kinematic_test_predictions.json \
  --output outputs/kinematic/test/kinematic_test_metrics.json
```

Para diagnóstico de desenvolvimento sem usar o próprio episódio no ajuste:

```bash
python scripts/segment_kinematic_subtasks.py \
  --root "$DATASET_ROOT" \
  --ground-truth "$GROUND_TRUTH" \
  --split "$SPLIT" --subset development --leave-one-out \
  --smoothing-s 1.0 --min-duration-s 2.5 \
  --output outputs/kinematic/development/kinematic_development_loo_predictions.json
```

### 4. Qwen em vocabulário aberto

`run_qwen_open_vocab.sh` usa o prompt padrão de planejamento do LeRobot. O Qwen pode escrever frases livres como `pick up the green cube`, `pull open the drawer` ou `place the cube in the drawer`; ele não é obrigado a emitir cinco frases.

Essa liberdade produz linguagem mais natural, mas também causa omissões de `move` e `open` e intervalos temporais pouco confiáveis. No método final, portanto, o Qwen aberto fornece **semântica**, não fronteiras.

```bash
export QWEN_OPEN_MANUAL45_ROOT="$DATA_ROOT/qwen_open_manual45"

./scripts/run_qwen_open_vocab.sh \
  --dataset-root "$DATASET_ROOT" \
  --out-root "$QWEN_OPEN_MANUAL45_ROOT" \
  --episodes '[0,5,7,11,16,21,24,35,38,40,45,49,55,58,63,67,70,75,81,84,87,89,91,94,96,98,99,100,101,107,113,125,135,142,147,150,155,158,161,163,166,167,169,171,249]' \
  --model-id "$QWEN_MODEL" \
  --api-base "$QWEN_API_BASE" \
  --camera-key "$CAMERA_KEY"

python scripts/export_lerobot_qwen_predictions.py \
  --root "$QWEN_OPEN_MANUAL45_ROOT" \
  --output outputs/qwen/open_vocab/manual45_raw.json

python scripts/evaluate_open_vocab_semantics.py \
  --ground-truth "$GROUND_TRUTH" \
  --qwen outputs/qwen/open_vocab/manual45_raw.json \
  --split "$SPLIT" --subset test \
  --output outputs/qwen/open_vocab/manual45_test_semantic_metrics.json
```

### 5. Híbrido aberto: ponto principal do pipeline

O híbrido aberto é chamado por:

```text
scripts/rebind_open_vocab_semantics.py
```

Ele recebe três elementos:

1. um JSON com cinco intervalos cinemáticos por episódio;
2. um JSON com frases livres exportadas do Qwen;
3. o encoder textual `google/siglip-base-patch16-224`.

O script converte as frases do Qwen e descrições canônicas em embeddings SigLIP. Depois, alinha frases às fases em ordem temporal. Se uma frase é semanticamente compatível, ela é preservada; se o Qwen não descreveu uma fase, o script usa fallback colorido a partir da fase cinemática e dos metadados da tarefa.

Os timestamps do Qwen não são usados como resposta final. Cada etapa produzida pelo híbrido contém fronteiras cinemáticas e campos de proveniência, como `qwen_open_vocab_text`, `semantic_cosine` e `canonical_semantic_target`.

Execute o híbrido nos 45 episódios manuais:

```bash
python scripts/segment_kinematic_subtasks.py \
  --root "$DATASET_ROOT" \
  --ground-truth "$GROUND_TRUTH" \
  --split "$SPLIT" \
  --episodes 0 5 7 11 16 21 24 35 38 40 45 49 55 58 63 67 70 75 81 84 87 89 91 94 96 98 99 100 101 107 113 125 135 142 147 150 155 158 161 163 166 167 169 171 249 \
  --smoothing-s 1.0 --min-duration-s 2.5 \
  --output outputs/kinematic/manual45/kinematic_manual45_predictions.json

python scripts/rebind_open_vocab_semantics.py \
  --kinematic outputs/kinematic/manual45/kinematic_manual45_predictions.json \
  --qwen outputs/qwen/open_vocab/manual45_raw.json \
  --output outputs/hybrid/open_vocab/manual45_siglip_kinematic.json
```

Avalie as fronteiras do híbrido no teste:

```bash
python scripts/evaluate_subtasks.py \
  --ground-truth "$GROUND_TRUTH" \
  --predictions outputs/hybrid/open_vocab/manual45_siglip_kinematic.json \
  --episodes 5 7 11 16 21 35 40 49 55 67 81 84 87 89 91 94 96 100 101 249 107 113 150 155 161 163 166 167 169 171 \
  --output outputs/hybrid/open_vocab/manual45_test_temporal_metrics.json
```

As métricas temporais do híbrido aberto devem coincidir com as do método cinemático quando ambos usam os mesmos intervalos. Isso é esperado e evidencia a divisão explícita de responsabilidades: a cinemática localiza ações; o Qwen fornece texto.

### 6. Híbrido aberto para o dataset completo

Para treinar o ACT, execute Qwen aberto sobre todos os episódios válidos, exporte as frases e gere os intervalos cinemáticos completos. Se houver várias execuções do Qwen, una os JSONs com `merge_qwen_prediction_exports.py`.

```bash
# 1. Qwen aberto nos 261 episódios válidos.
export QWEN_OPEN_ALL_ROOT="$DATA_ROOT/qwen_open_all"
export QWEN_ALL_EPISODES="$(python -c 'import json; print(json.dumps([i for i in range(264) if i not in {182, 194, 230}]))')"
export KINEMATIC_ALL_EPISODES="$(python -c 'print(*[i for i in range(264) if i not in {182, 194, 230}])')"
./scripts/run_qwen_open_vocab.sh \
  --dataset-root "$DATASET_ROOT" \
  --out-root "$QWEN_OPEN_ALL_ROOT" \
  --episodes "$QWEN_ALL_EPISODES" \
  --model-id "$QWEN_MODEL" \
  --api-base "$QWEN_API_BASE" \
  --camera-key "$CAMERA_KEY"

python scripts/export_lerobot_qwen_predictions.py \
  --root "$QWEN_OPEN_ALL_ROOT" \
  --output outputs/qwen/open_vocab/all_raw.json

# 2. Fronteiras cinemáticas para os 261 episódios válidos.
python scripts/segment_kinematic_subtasks.py \
  --root "$DATASET_ROOT" --ground-truth "$GROUND_TRUTH" --split "$SPLIT" \
  --episodes $KINEMATIC_ALL_EPISODES \
  --smoothing-s 1.0 --min-duration-s 2.5 \
  --output outputs/kinematic/all/kinematic_all_predictions.json

# 3. Híbrido completo. Episódios sem resposta do Qwen recebem fallback.
python scripts/rebind_open_vocab_semantics.py \
  --kinematic outputs/kinematic/all/kinematic_all_predictions.json \
  --qwen outputs/qwen/open_vocab/all_raw.json \
  --include-unannotated-kinematic \
  --output outputs/hybrid/open_vocab/all_siglip_kinematic.json
```

## Onde cada método é implementado

| Componente | Código | Responsabilidade |
|---|---|---|
| Qwen fechado | `run_qwen_annotation.sh` | Executa LeRobot Annotate com prompt fixo de ações e cores |
| Qwen aberto | `run_qwen_open_vocab.sh` | Executa LeRobot Annotate com o prompt padrão aberto |
| Exportação | `export_lerobot_qwen_predictions.py` | Converte anotações do dataset copiado para JSON de predições |
| Cinemático | `segment_kinematic_subtasks.py` | Extrai sinais do robô e produz cinco intervalos ordenados |
| Híbrido aberto | `rebind_open_vocab_semantics.py` | Alinha texto do Qwen a fases cinemáticas via SigLIP e cria fallback |
| Temporal | `evaluate_subtasks.py` | Calcula tIoU, Boundary MAE e F1 fechado quando aplicável |
| Semântico | `evaluate_open_vocab_semantics.py` | Calcula cosseno SigLIP, cobertura e cosseno conservador |
| Manifesto | `build_open_vocab_language_manifest.py` | Cria o JSONL consumido pelo dataloader do ACT |

## Métricas

**tIoU** mede sobreposição temporal entre intervalo previsto e manual:

```text
tIoU = |I_previsto ∩ I_manual| / |I_previsto ∪ I_manual|
```

Quanto maior, melhor.

**Boundary MAE** mede o erro absoluto médio dos tempos de transição, em segundos:

```text
Boundary MAE = (1 / B) Σ |b_previsto,i − b_manual,i|
```

Quanto menor, melhor.

No vocabulário fechado, F1 avalia rótulos estruturados. No vocabulário aberto, as frases são comparadas com cosseno entre embeddings SigLIP:

```text
cos(e_Qwen, e_manual) = (e_Qwen · e_manual) / (||e_Qwen|| ||e_manual||)
```

- **Cobertura semântica:** fração de fases manuais para as quais Qwen emitiu frase alinhada.
- **Cosseno conservador:** atribui zero às fases omitidas; combina cobertura e qualidade textual.

O F1 genérico do método cinemático pode ser `1.0` por construção, pois ele sempre emite as cinco fases. As métricas temporalmente informativas para esse método são tIoU e Boundary MAE.

## Figuras, tabelas e vídeos

As figuras finais consideram quatro métodos: Qwen fechado, Qwen aberto, cinemático e híbrido aberto. O híbrido fechado não faz parte da comparação final.

```bash
python scripts/plot_method_comparison.py \
  --qwen-closed outputs/qwen/qwen7b_test_metrics.json \
  --qwen-open outputs/qwen/open_vocab/manual45_test_raw_temporal_metrics.json \
  --kinematic outputs/kinematic/test/kinematic_test_metrics.json \
  --hybrid-open outputs/hybrid/open_vocab/manual45_test_temporal_metrics.json \
  --open-semantic outputs/qwen/open_vocab/manual45_test_semantic_metrics.json \
  --manifest datasets_annotation/language_manifests/open_vocab/all261_open_vocab_hybrid_v1.jsonl \
  --output-dir outputs/figures/method_comparison_matplotlib
```

Para a comparação visual de episódios:

```bash
python scripts/plot_all_method_timelines.py \
  --ground-truth "$GROUND_TRUTH" \
  --qwen-closed outputs/qwen/qwen7b_test_predictions.json \
  --qwen-open outputs/qwen/open_vocab/manual45_raw.json \
  --kinematic outputs/kinematic/test/kinematic_test_predictions.json \
  --hybrid-open outputs/hybrid/open_vocab/manual45_siglip_kinematic.json \
  --episodes 5 55 107 \
  --output outputs/figures/method_comparison_matplotlib/05_qualitative_all_methods_timelines
```


```bash
# Vídeo do Qwen aberto com seus tempos originais.
python scripts/visualize_qwen_subtasks.py \
  --root "$QWEN_OPEN_MANUAL45_ROOT" --episode 5 --camera-key "$CAMERA_KEY" \
  --output outputs/videos/qwen/episode_005_open_vocab_qwen.mp4

# Vídeo do híbrido com fronteiras cinemáticas e texto Qwen/fallback.
python scripts/visualize_subtask_predictions.py \
  --root "$DATASET_ROOT" \
  --predictions outputs/hybrid/open_vocab/all_siglip_kinematic.json \
  --episode 5 --camera-key "$CAMERA_KEY" \
  --output outputs/videos/hybrid/episode_005_open_vocab_hybrid.mp4
```

## Manifesto linguístico e ACT

O manifesto é a interface entre rotulação e treinamento. Ele contém uma linha JSON por episódio, com tarefa global, cores, intervalos, texto da subtarefa, fonte da linguagem e similaridade semântica.

```bash
python scripts/build_open_vocab_language_manifest.py \
  --dataset-root "$DATASET_ROOT" \
  --predictions outputs/hybrid/open_vocab/all_siglip_kinematic.json \
  --output datasets_annotation/language_manifests/open_vocab/all_open_vocab_hybrid_v1.jsonl \
  --vocabulary-output datasets_annotation/language_manifests/open_vocab/all_open_vocab_hybrid_v1_vocabulary.json
```

Exemplo conceitual de segmento:

```json
{
  "generic_phase": "move",
  "start_frame": 368,
  "end_frame": 644,
  "subtask_text": "move the green cube to the green drawer",
  "language_source": "qwen_open_vocab",
  "semantic_cosine": 0.96
}
```

Durante o treinamento, o dataloader localiza o segmento que contém o frame inicial de cada chunk, passa `subtask_text` pelo encoder textual SigLIP e fornece o embedding ao ACT:

```text
(imagem, estado do robô, embedding SigLIP da subtarefa) → ações previstas
```

Na primeira versão do dataloader, recomenda-se descartar chunks que atravessem uma fronteira de subtarefa. Isso garante que cada chunk seja condicionado por uma única instrução local.
