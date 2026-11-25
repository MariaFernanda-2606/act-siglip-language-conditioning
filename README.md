# ACT SigLIP: anotação de subtarefas e ACT condicionado por linguagem

Este repositório cobre a tarefa de pegar um cubo colorido e colocá-lo na gaveta da cor correspondente, no robô bimanual Trossen AI Stationary, em duas partes:

1. **Anotação** ([Parte I](#parte-i--anotação-de-subtarefas)): gera um **manifesto linguístico** que associa cada intervalo de uma demonstração a uma subtarefa em linguagem natural.
2. **ACT condicionado por linguagem** ([Parte II](#parte-ii--act-condicionado-por-linguagem-act_siglip)): o texto da subtarefa vira um embedding SigLIP que condiciona o ACT (`act_siglip`). No robô, um planejador divide o comando completo nas cinco subtarefas e avança entre elas pela cabeça de conclusão do modelo.

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
├── configs/
│   ├── evaluation_split.json                    # 15 episódios de desenvolvimento, 30 de teste
│   └── train/                                   # configs dos 4 treinos do modelo final (ACT)
├── datasets_annotation/
│   ├── evaluation_ground_truth/                 # referência manual
│   └── language_manifests/open_vocab/           # manifesto final para o ACT
├── lerobot_policy_act_siglip/                   # plugin LeRobot da política act_siglip
├── prompts/cube_drawer_subtasks.txt             # prompt do Qwen fechado
├── scripts/
│   ├── run_qwen_annotation.sh                   # Qwen fechado
│   ├── run_qwen_open_vocab.sh                   # Qwen aberto
│   ├── segment_kinematic_subtasks.py            # segmentação cinemática (+ refinamento por eventos)
│   ├── rebind_open_vocab_semantics.py           # híbrido aberto
│   ├── build_open_vocab_language_manifest.py    # manifesto JSONL
│   ├── evaluate_subtasks.py                     # métricas temporais
│   ├── evaluate_open_vocab_semantics.py         # métricas semânticas
│   ├── plot_method_comparison.py                # figuras de comparação
│   ├── plot_all_method_timelines.py             # timelines qualitativas
│   ├── encode_subtask_vocabulary.py             # ACT: tabela de embeddings SigLIP das instruções
│   ├── build_frame_subtask_index.py             # ACT: índice frame -> instrução
│   ├── build_grasp_targets.py                   # ACT: alvos da cabeça auxiliar de pegada
│   ├── train_act_siglip.sh                      # ACT: treino do modelo final (4 etapas)
│   ├── evaluate_act_siglip_offline.py           # ACT: avaliação offline dos checkpoints
│   ├── run_act_siglip_robot.py                  # ACT: execução e avaliação no robô
│   └── cube_area_overlay.py                     # ACT: área e layouts dos cubos nos testes
└── outputs/                                     # resultados gerados localmente (fora do git)
```

## Dataset final

O modelo final foi treinado com **`place_the_cubes_in_the_drawers_v4`** (LeRobot v3.0, 300 episódios, 324.541 frames, 30 fps, 4 câmeras, 13 GB), público no Hugging Face:

**https://huggingface.co/datasets/MariaFPG/place_the_cubes_in_the_drawers_v4**

Para baixar para o caminho que os configs de treino esperam:

```bash
export DATA_ROOT=~/.cache/huggingface/lerobot/MariaFPG
python -c "from huggingface_hub import snapshot_download; snapshot_download('MariaFPG/place_the_cubes_in_the_drawers_v4', repo_type='dataset', local_dir='$DATA_ROOT/place_the_cubes_in_the_drawers_v4')"
```

| Episódios | Origem |
|---|---|
| 0–209 | `MariaFPG/place_the_cubes_in_the_drawers_v3_annotated` (idênticos) |
| 210–244 | vermelho: primeiros 35 de `MariaFPG/place_the_red_cube_in_the_red_drawer` |
| 245–279 | amarelo: primeiros 35 de `MariaFPG/place_the_yellow_cube_in_the_yellow_drawer` |
| 280–299 | verde: primeiros 20 de `MariaFPG/place_the_green_cube_in_the_green_drawer` |

Os episódios 210–299 vieram dos datasets por cor (LeRobot v2.1), convertidos para v3.0 e agregados ao v3_annotated. As etapas 1–3 do treino usam o `v3_annotated`, que está **privado** no Hugging Face; a etapa 4 (modelo final) usa só o `v4`.

As anotações usadas no treino ficam dentro do próprio dataset, em `annotations/`:

| Arquivo | Conteúdo |
|---|---|
| `open_vocab_hybrid/all300_open_vocab_hybrid_v4.jsonl` | manifesto: 5 intervalos por episódio com o texto da subtarefa |
| `language_conditioning/siglip_text_table_v3.npz` | embeddings SigLIP de cada instrução |
| `language_conditioning/frame_subtask_index_v4.npz` | instrução e fase de cada frame |
| `language_conditioning/grasp_targets_v4.npz` | alvo da cabeça auxiliar de pegada |
| `evaluation/evaluation_split_48.json` | 15 episódios de desenvolvimento e 33 de teste (fora do treino) |

Treino: 236 episódios = 300 − 48 (avaliação) − 16 excluídos por segmentação implausível (220, 225, 228, 229, 230, 235, 236, 245, 248, 253, 261, 262, 274, 288, 291, 299). A lista exata está em `dataset.episodes` de [`configs/train/04_progress_v4.json`](configs/train/04_progress_v4.json).

# Parte I — Anotação de subtarefas


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

Durante o treinamento, cada frame recebe o embedding SigLIP do texto da subtarefa que o contém, e o ACT aprende:

```text
(imagem, estado do robô, embedding SigLIP da subtarefa) → ações previstas
```

O uso desse manifesto no treino está descrito na Parte II.

# Parte II — ACT condicionado por linguagem (`act_siglip`)

## Visão geral

```text
comando ("pick up the red cube and place it inside the red drawer")
   │
   ▼
planejador determinístico ── cor → 5 subtarefas canônicas, em ordem fixa
   │  subtarefa atual (ex.: "open the red drawer")
   ▼
SigLIP (texto) ──> ACT (imagens + estado + texto) ──> chunk de 100 ações
                     │
                     └── cabeça de conclusão/progresso ──> planejador avança de subtarefa
```

O plugin [`lerobot_policy_act_siglip`](lerobot_policy_act_siglip/) registra a política `act_siglip` no LeRobot. Em relação ao ACT original, ele adiciona:
- condicionamento pelo texto (token de linguagem e FiLM nas camadas 2–4 do ResNet);
- cabeça de conclusão e níveis de progresso da subtarefa, usados pelo planejador;
- cabeça auxiliar de pegada;
- dropout do estado;
- máscaras de imagem: faixas da `cam_high` com pessoas e fundo variável.

O planejador está em [`planner.py`](lerobot_policy_act_siglip/src/lerobot_policy_act_siglip/planner.py).

## Ambiente

O treino e o robô usam LeRobot **0.5.2** (lê datasets v3.0), num ambiente separado do de anotação:

```bash
conda activate act_siglip            # Python com lerobot==0.5.2, transformers, sentencepiece, protobuf
pip install -e lerobot_policy_act_siglip
```

Rode os comandos a partir da raiz deste repositório. Não rode dentro de uma pasta que contenha outro pacote `lerobot`, porque ele sobrepõe o instalado. Para o robô também são necessários `trossen_arm==1.8.4` (mesma versão do firmware) e `pyrealsense2`.

## Preparação dos dados para o ACT

Os arquivos usados pelo modelo final já estão em `annotations/` do dataset v4 (ver [Dataset final](#dataset-final)). Para gerá-los a partir de um manifesto:

```bash
export DATA_ROOT=~/.cache/huggingface/lerobot/MariaFPG
export DS=$DATA_ROOT/place_the_cubes_in_the_drawers_v4
export LC=$DS/annotations/language_conditioning
export MANIFEST=$DS/annotations/open_vocab_hybrid/all300_open_vocab_hybrid_v4.jsonl

# 1. embeddings SigLIP de cada instrução única do manifesto
python scripts/encode_subtask_vocabulary.py --manifest $MANIFEST --output $LC/siglip_text_table_v3.npz

# 2. instrução e fase de cada frame do dataset
python scripts/build_frame_subtask_index.py --dataset-root $DS --manifest $MANIFEST \
  --embeddings $LC/siglip_text_table_v3.npz --output $LC/frame_subtask_index_v4.npz

# 3. alvos da cabeça de pegada (juntas do braço esquerdo no fim do pick)
python scripts/build_grasp_targets.py --root $DS --frame-index $LC/frame_subtask_index_v4.npz \
  --out $LC/grasp_targets_v4.npz
```

O manifesto v4 usa as fronteiras de `segment_kinematic_subtasks.py` com refinamento por eventos e o texto canônico de cada fase. No conjunto de teste manual, o tIoU foi 0,792 e o Boundary MAE 0,99 s.

## Treinamento

O modelo final (**v7**, `act_siglip_progress_v4`, checkpoint `015000`) está público no Hugging Face: **https://huggingface.co/MariaFPG/act_siglip_v7**

```bash
huggingface-cli download MariaFPG/act_siglip_v7 --local-dir models/act_siglip_v7   # depois: --policy-path models/act_siglip_v7
```

Ele vem de 4 treinos em sequência. Cada treino parte do checkpoint do anterior:

| Etapa | Config | Passos | Parte de | Dataset | Mudança |
|---|---|---:|---|---|---|
| 1 | [`01_mask_aug_pilot`](configs/train/01_mask_aug_pilot.json) | 50k | do zero | v3_annotated | máscaras de imagem, augmentation (brilho, contraste, nitidez, afim) |
| 2 | [`02_open_ft`](configs/train/02_open_ft.json) | 10k | 1 → `050000` | v3_annotated | sem augmentation afim; mais peso em `open` |
| 3 | [`03_place_ft`](configs/train/03_place_ft.json) | 10k | 2 → `010000` | v3_annotated | mais peso em `place` e `close` |
| 4 | [`04_progress_v4`](configs/train/04_progress_v4.json) | 15k | 3 → `007500` | v4 | +90 demonstrações, níveis de progresso |

```bash
export DATA_ROOT=~/.cache/huggingface/lerobot/MariaFPG   # pasta com _v3_annotated/ e _v4/
./scripts/train_act_siglip.sh                            # as 4 etapas
./scripts/train_act_siglip.sh 04_progress_v4             # só uma etapa (a anterior precisa existir)
```

O script troca `${DATA_ROOT}` nos configs, roda `lerobot-train --config_path=...` com `ACCELERATE_MIXED_PRECISION=bf16` e salva em `outputs/train/<nome>/`, com o log em `outputs/train/<nome>.log`. Na GPU usada no projeto cada passo levou ~0,8 s (batch 8), ou seja, ~3,5 h para 15k passos.

## Avaliação offline

Mede, em episódios fora do treino, o erro das ações, a cabeça de conclusão e o teste de troca de cor: com a cor trocada no texto, as ações devem mudar, principalmente no `pick`.

```bash
python scripts/evaluate_act_siglip_offline.py \
  --run-dir outputs/train/act_siglip_progress_v4 \
  --root $DATA_ROOT/place_the_cubes_in_the_drawers_v4 \
  --frame-index $DATA_ROOT/place_the_cubes_in_the_drawers_v4/annotations/language_conditioning/frame_subtask_index_v4.npz \
  --split development --stride 10 --out outputs/eval/act_siglip_progress_v4_dev
```

## Execução no robô

[`run_act_siglip_robot.py`](scripts/run_act_siglip_robot.py) controla o Trossen AI Stationary diretamente: braços em modo de posição e 4 câmeras RealSense. Ele tem quatro modos, para usar nesta ordem:

| Modo | O que faz |
|---|---|
| `check` | só câmeras e braços: vai para home e salva imagens ao lado das do dataset |
| `dry-run` | roda a política, mas não envia as ações |
| `run` | executa a tarefa |
| `eval` | N tentativas seguidas do mesmo comando, com rótulo do operador, métricas e gravação opcional como dataset LeRobot |

Comando com as opções usadas nos testes do modelo final:

```bash
python scripts/run_act_siglip_robot.py --mode run \
  --policy-path outputs/train/act_siglip_progress_v4/checkpoints/015000/pretrained_model \
  --replan-every 30 --crossfade 20 --crossfade-joints-only --fps 20 --done-every 5 --patience 5 \
  --close-after-release 1.5 --release-timeout 4 \
  --command "pick up the green cube and place it inside the green drawer"
```

| Opção | Função |
|---|---|
| `--replan-every 30 --crossfade 20` | novo chunk a cada 30 frames, misturado ao anterior por 20 frames (sigmoide) |
| `--crossfade-joints-only` | as garras seguem o chunk novo sem mistura, o que preserva o pulso de soltura |
| `--done-every 5 --patience 5` | avalia a conclusão a cada 5 frames; avança após 5 notas ≥ 0,5 |
| `--close-after-release 1.5` | passa para `close` 1,5 s depois que a garra esquerda solta o cubo |
| `--release-timeout 4` | no `place`, abre a garra se o braço ficar 4 s parado sem soltar |
| `--smooth-chunk 1` | suaviza cada chunk no tempo (juntas, sem atraso); reduz os trancos |

Durante a execução, digite no terminal e aperte Enter: `n` força a próxima subtarefa; `q` encerra.

### Avaliação no robô

```bash
python scripts/run_act_siglip_robot.py --mode eval --episodes 15 \
  --policy-path outputs/train/act_siglip_progress_v4/checkpoints/015000/pretrained_model \
  --replan-every 30 --crossfade 20 --crossfade-joints-only --fps 20 --done-every 5 --patience 5 --log-chunks \
  --close-after-release 1.5 --release-timeout 4 \
  --command "pick up the green cube and place it inside the green drawer" \
  --eval-name v7_green --hf-repo MariaFPG/eval_v7_green
```

Funcionamento do modo `eval`:
- **Cena:** antes de cada tentativa, o script confere a posição dos cubos pela `cam_high` e alterna 6 layouts fixos. Para isso, lê `docs/teste_robo/area_cubos.json`; sem esse arquivo, a checagem é pulada.
- **Rótulo:** ao fim de cada tentativa, o operador marca até onde o robô chegou: 0–5, onde 5 = gaveta fechada (sucesso). A opção `p` marca que o pick falhou e foi pulado com `n`; nesse caso as fases seguintes são avaliadas separadamente.
- **Resultados:** ficam em `outputs/robot/eval/<eval-name>/` (`trials.csv`, `summary.json`, rollouts).
- **Retomada:** rodar de novo com o mesmo `--eval-name` continua a avaliação de onde parou.
- **Gravação:** com `--hf-repo`, as tentativas são gravadas como dataset LeRobot. No fim, o script pergunta antes de enviar ao Hugging Face (privado por padrão).
