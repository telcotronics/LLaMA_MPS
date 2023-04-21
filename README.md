# LLaMA_MPS
Run LLaMA (and Stanford-Alpaca) inference on Apple Silicon GPUs.

![Demo](demo.gif)

Como puede ver, a diferencia de otros LLM, LLaMA no está sesgado de ninguna manera 😄

### Initial setup steps

**1. Clone this repo**

`git clone https://github.com/jankais3r/LLaMA_MPS`

**2. Instalar dependencias de Python**

```bash
cd LLaMA_MPS
pip3 install virtualenv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

### Configuración específica de LLaMA

**3. [Download the model weights](https://github.com/facebookresearch/llama/pull/73/files#diff-b335630551682c19a781afebcf4d07bf978fb1f8ac04c6bf87428ed5106870f5R4) and put them into a folder called** `models` (e.g., `LLaMA_MPS/models/7B`)

**4. _(Optional)_ Reshard the model weights (13B/30B/65B)**

Dado que estamos ejecutando la inferencia en una sola GPU, debemos fusionar los pesos de los modelos más grandes en un solo archivo.

```bash
mv models/13B models/13B_orig
mkdir models/13B
python3 reshard.py 1 models/13B_orig models/13B
```

**5. Ejecutar la inferencia**

`python3 chat.py --ckpt_dir models/13B --tokenizer_path models/tokenizer.model --max_batch_size 8 --max_seq_len 256`

Los pasos anteriores le permitirán ejecutar la inferencia en el modelo LLaMA sin procesar en un modo de 'autocompletar'.

Si desea probar el modo de 'instrucción-respuesta' similar a ChatGPT usando los pesos ajustados de [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), continúe con la configuración con los siguientes pasos :

### Configuración específica de Alpaca

![Alpaca demo](alpaca.gif)

**3. Descargue los pesos ajustados (disponibles para 7B/13B)**

```bash
python3 export_state_dict_checkpoint.py 7B
python3 clean_hf_cache.py
```

**4. Ejecutar la inferencia**

`python3 chat.py --ckpt_dir models/7B-alpaca --tokenizer_path models/tokenizer.model --max_batch_size 8 --max_seq_len 256`

### Requisitos de memoria

|modelo|inicio memoria durante la inferencia|Pico de memoria durante la conversión checkpoint|Pico de memoria durante la fragmentación|
| Model | Starting memory during inference | Peak memory during checkpoint conversion | Peak memory during resharding |
| ------------- | ------------- | ------------- | ------------- |
| 7B | 16 GB | 14 GB | N/A |
| 13B | 32 GB | 37 GB | 45 GB |
| 30B | 66 GB | 76 GB | 125 GB |
| 65B | ?? GB | ?? GB | ?? GB |

**Especificaciones mínimas por modelo (slow due to swapping):**

* 7B - 16 GB RAM
* 13B - 32 GB RAM
* 30B - 64 GB RAM
* 65B - needs testing

**Especificaciones recomendadas por modelo:**

* 7B - 24 GB RAM
* 13B - 48 GB RAM
* 30B - 96 GB RAM
* 65B - needs testing

### Parámetros para experimentar
**- max_batch_size**

Si tiene memoria libre (por ejemplo, cuando ejecuta el modelo 13B en una Mac de 64 GB), puede aumentar el tamaño del lote usando el argumento `--max_batch_size=32`. El valor predeterminado es `1`.

**- max_seq_len**

Para aumentar/disminuir la longitud máxima del texto generado, utilice el argumento `--max_seq_len=256`. El valor predeterminado es `512`.

**- use_repetition_penalty**

El script de ejemplo penaliza al modelo por generar un contenido repetitivo. Esto debería conducir a una salida de mayor calidad, pero ralentiza ligeramente la inferencia. Ejecute el script con el argumento `--use_repetition_penalty=False` para desactivar el algoritmo de penalización.

### Alternatives

La mejor alternativa a LLaMA_MPS para los usuarios de Apple Silicon es [llama.cpp](https://github.com/ggerganov/llama.cpp), que es una reimplementación de C/C++ que ejecuta la inferencia únicamente en la parte de la CPU de el SoC. Debido a que el código C compilado es mucho más rápido que Python, en realidad puede superar esta implementación de MPS en velocidad, sin embargo, a costa de una potencia y una eficiencia térmica mucho peores.
See the below comparison when deciding which implementation better fits your use case.

| Implementation | Total run time - 256 tokens | Tokens/s | Peak memory use | Peak SoC temperature | Peak SoC Power consumption | Tokens per 1 Wh |
| -------------- | ------------------------------- | ----------------------------- | ------------- | ------------------------- | ------------------------------ | --------------------------- |
| LLAMA_MPS (13B fp16) | 75 s | 3.41 | 30 GB | 79 °C | 10 W | 1,228.80 |
| llama.cpp (13B fp16) | 70 s | 3.66 | 25 GB | 106 °C | 35 W | 376.16 |

### Credits

- facebookresearch ([original code](https://github.com/facebookresearch/llama))
- markasoftware ([cpu optimizations](https://github.com/markasoftware/llama-cpu))
- remixer-dec ([mps optimizations](https://github.com/remixer-dec/llama-mps))
- venuatu ([continuous token printing](https://github.com/venuatu/llama/commit/25c84973f71877677547453dab77eeaea9a86376) / [loading optimizations](https://github.com/venuatu/llama/commit/0d2bb5a552114b69db588175edd3e55303f029be))
- benob ([reshard script](https://gist.github.com/benob/4850a0210b01672175942203aa36d300))
- tloen ([repetition penalty](https://github.com/tloen/llama-int8) / [LoRA merge script](https://github.com/tloen/alpaca-lora/blob/main/export_state_dict_checkpoint.py))
