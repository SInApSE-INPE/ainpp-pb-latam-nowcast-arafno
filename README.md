# ARAFNO — inferência operacional GSMaP NOW

Pipeline independente para aquisição, montagem do ARAFNO, carregamento estrito dos
pesos, processamento e previsão autorregressiva em GeoTIFF. Usa **GSMaP NOW**, conforme
a escolha operacional, seis observações horárias consecutivas e, por padrão, seis
horas de previsão. Não precisa de observações futuras, Zarr, Cartopy ou matplotlib.

Aquisição, processamento, exportação e agendamento estão em
`src/arafno/_runtime/`, incluído no próprio pacote. A pasta funciona sem as
demais pastas de modelos e sem `../shared/nowcasting_runtime`. Após atualizar
uma instalação existente, execute `uv sync --locked`.

## Organização

```text
01_ARAFNO/
├── pyproject.toml / uv.lock     # pacote instalável e dependências reproduzíveis
├── config/latam.json           # domínio, produto, orientação, faltantes e artefatos
├── src/arafno/
│   ├── model.py                # arquitetura, carregamento e rollout autorregressivo
│   ├── _runtime/               # aquisição, processamento, GeoTIFF e agendamento privados
│   ├── acquisition.py          # arquivos locais, FTP, cache e seleção temporal
│   ├── processing.py           # leitura binária, recorte e normalização
│   ├── export.py               # escrita dos GeoTIFFs
│   ├── artifacts.py            # integridade do pacote de pesos
│   ├── config.py               # contrato de configuração
│   ├── formats.py              # constantes únicas do formato GSMaP e saída
│   ├── scheduler.py            # consulta agendada, lock e deduplicação de emissões
│   └── entrypoint.py           # CLI e orquestração
├── artifacts/v1/               # pesos, média, desvio e manifesto SHA-256
├── reference/                  # três scripts originais preservados
├── tests/                      # contratos temporais, espaciais e integração
├── deploy/                     # timer systemd, gerador e instruções de instalação
├── data/raw/                   # entradas locais opcionais, não apagadas pelo pipeline
└── outputs/                    # previsões geradas, fora do controle de versão
```

## Instalação e execução

Cada parâmetro tem uma única fonte no código operacional:

| Parâmetros | Fonte |
|---|---|
| Limites do domínio, produto, diretório remoto, orientação, artefatos e política de faltantes | `config/latam.json`; campos obrigatórios, sem padrões duplicados em Python |
| Resolução nativa, extensão e formato binário global, CRS e unidade | `src/arafno/formats.py`; dimensões e tamanho em bytes são derivados |
| Canais de entrada, embedding, profundidade e patch | Checkpoint; o construtor não define padrões alternativos |
| Fonte local/FTP, pastas, horizonte, dispositivo e threads | Argumentos da CLI; padrões definidos somente no parser |
| Credenciais | Variáveis de ambiente |
| Timeout e retentativas FTP | Seção `ftp` de `config/latam.json` |
| Frequência de consulta agendada | `OnCalendar` em `deploy/arafno-now.timer` |

A resolução de 0,1° é parte do contrato do binário GSMaP, não uma opção de
reamostragem. Por isso, `resolution` foi removido do JSON; cópias antigas da
configuração devem remover esse campo. A resolução efetiva é registrada no
manifesto de cada execução. Os valores de arquitetura no manifesto de artefatos
são descritivos: a montagem lê os pesos. Os scripts em `reference/` são originais
preservados e não são importados pelo pipeline.

Python 3.11 ou superior. Execute a partir desta pasta:

```bash
uv sync --locked --group dev
uv run --locked arafno --help
```

O lock usa PyTorch CPU para uma instalação portátil. Em um ambiente CUDA, instale a
distribuição de PyTorch compatível com o driver e depois `pip install .`; o pacote
aceita o PyTorch já instalado. Execute `arafno --device cuda ...` nesse ambiente.
Não use `uv sync` com o lock CPU sobre esse ambiente CUDA.

Leitura local dos binários globais originais, ainda em mm/h:

```bash
uv run --locked arafno --config config/latam.json \
  --source local --input-dir data/raw/now \
  --run-time 2026-10-19T12:00:00Z --steps 6 --device cpu
```

Nesse exemplo são necessários os arquivos `gsmap_now.20261019.0700.dat.gz` até
`gsmap_now.20261019.1200.dat.gz`. Sem `--run-time`, usa a observação mais recente
e exige a janela completa que termina nela. Não recua silenciosamente para uma
janela antiga se houver lacunas. O diretório local é pesquisado recursivamente;
duplicatas de horário são recusadas.

Também aceita a nomenclatura NOW com intervalo horário explícito, por exemplo
`gsmap_now.20261019.1200_1300.dat.gz`, usando o horário inicial como rótulo temporal
do produto. Arquivos de meia hora (`HH30`) não entram na sequência horária.

Acesso operacional FTP:

```bash
export FTP_JAXA_HOST=hokusai.eorc.jaxa.jp
export FTP_JAXA_USER='seu_usuario'
export FTP_JAXA_PASSWORD='sua_senha'
uv run --locked arafno --config config/latam.json --source ftp \
  --output-dir outputs --device auto
```

Por padrão, os downloads FTP ficam no diretório temporário do sistema, em
`nowcasting-<UID>/arafno`, e expiram após 24 horas. `--cache-ttl-hours 6`
altera esse prazo; `--cache-dir` permite escolher outro cache gerenciado.
A limpeza ocorre em cada execução FTP ou consulta agendada, inclusive quando
a previsão já existe. Em uma execução manual isolada, a remoção ocorrerá na
próxima execução ou pela política de limpeza do sistema. Dados passados via
`--source local --input-dir` nunca são removidos pelo pipeline.

As credenciais são lidas somente no modo FTP, não são gravadas nos manifestos e não
devem entrar no repositório. O serviço usa o FTP da referência. Há timeout,
até três tentativas, download temporário e validação do gzip antes de publicar o
cache. Cache inválido é baixado novamente. Sem horário explícito, o modo FTP recusa
dados futuros ou com mais de `--max-age-hours` (padrão: 12 horas).

Para reprocessar um arquivo histórico, configure `remote_directory` com o diretório
de arquivo disponibilizado pelo provedor. Templates `%Y/%m/%d` são expandidos para
todos os dias da janela, inclusive na virada de dia, e exigem `--run-time`.
`/now/latest` só contém arquivos recentes; informar uma data antiga não cria acesso
ao histórico. Os caminhos da configuração são relativos ao JSON; caminhos da CLI
são relativos ao diretório de execução.

## Contrato científico e espacial

Para operação contínua, siga [Agendamento com systemd](deploy/README.md). O modo
`--source ftp --scheduled` faz uma consulta idempotente: o timer consulta a cada
cinco minutos, e a inferência só é executada para uma nova emissão horária.
Ele exige uma janela completa, reutiliza o cache e mantém retentativas para falhas
de transporte e arquivos ainda incompletos. As unidades precisam ser instaladas
e ativadas na máquina de operação; não estão ativas por terem sido geradas.


- Pesos inspecionados: entrada `[1, 6, 880, 970]`, embedding 256, profundidade 8,
  patches 10 × 10 e saída de uma hora. As dimensões são reconstruídas do checkpoint,
  comparadas com o domínio e carregadas com `strict=True` e `weights_only=True`.
- O leitor usa float32 little-endian, grade global 1200 × 3600, longitudes originais
  0–360°, giro para −180–180° e recorte de bordas `[-120, -55, -23, 33]`.
  A grade final tem 0,1°, centro inicial `(-119,95; 32,95)` e norte no topo.
- Normalização: `(log1p(chuva) - média) / desvio`, com os dois `.npy` fornecidos.
  A saída usa `max(expm1(previsão * desvio + média), 0)`. O próximo passo recebe a
  previsão normalizada; não se realimenta a chuva desnormalizada ou truncada.
- Os GeoTIFFs contêm **taxa de precipitação em mm/h**, não acumulado. O tempo de
  referência é o timestamp do último arquivo de entrada; `valid_time = run_time +
  lead_hours`. Os timestamps preservam a convenção do produto de origem e não são
  substituídos pelo relógio da execução.
- Valores finitos menores que zero, incluindo −99, são convertidos em **zero mm/h
  antes da normalização**. Esses pixels não contam como faltantes e não geram
  NoData. Valores negativos no espaço normalizado são válidos e não são truncados.
- NaN e infinitos de entrada continuam sendo faltantes. A configuração operacional
  usa `missing_policy: "normalized_zero"`, preenchendo com zero no espaço
  normalizado, como o tratamento de NaN na inferência de referência. Isso não
  equivale a zero de chuva. A opção `error` interrompe se houver qualquer faltante;
  `zero_rain` preenche zero mm/h antes da normalização. Nas opções de preenchimento, pixels
  com qualquer faltante na janela recebem NoData em todas as previsões. Cada quadro
  também está sujeito a `max_missing_fraction` (padrão 5%). Essa máscara não mede
  a propagação da incerteza pelo modelo, cujos filtros têm alcance espacial global.

**Validações científicas pendentes:** a inferência de referência utiliza NRT,
enquanto a operação solicitada usa NOW. Isso muda o produto de entrada e requer
avaliação de desempenho em NOW. A orientação espacial do treinamento não está
registrada nos pesos: `model_latitude_order: "north_to_south"` segue a leitura do
script operacional, mas precisa ser comparada com uma amostra real de treinamento.
Se ela for sul→norte, altere para `south_to_north`; o pipeline inverte a entrada
e reconverte a saída para norte no topo. O manifesto registra essa verificação
como pendente. Os limites geográficos vieram dos scripts; os pesos confirmam
somente o tamanho da grade, não os limites. Aumentar `--steps` é possível, mas não
comprova habilidade preditiva além do horizonte avaliado originalmente.

Formato binário e coordenadas do provedor:
[documentação oficial GSMaP NOW](https://sharaku.eorc.jaxa.jp/GSMaP/document/DataFormatDescription_NOW.pdf)
e [GSMaP NRT](https://sharaku.eorc.jaxa.jp/GSMaP/document/new/DataFormatDescription_NRT8.pdf).

## Saída e rastreabilidade

```text
outputs/arafno/gsmap_now/2026/10/19/1200Z/
├── arafno_20261019T1200Z_f001_20261019T1300Z.tif
├── ...
├── arafno_20261019T1200Z_f006_20261019T1800Z.tif
└── manifest.json
```

Cada arquivo tem uma banda float32, EPSG:4326, pixels de área, NoData −9999,
blocos 256 × 256 e compressão DEFLATE. Tags incluem modelo, versão, checksum dos
pesos, produto, emissão, validade, antecedência e unidade. O manifesto acrescenta
hashes das entradas e saídas, configuração, normalização, faltantes e versões de
bibliotecas. A pasta da execução é publicada por rename no mesmo filesystem
somente depois que todas as saídas estão completas. Falhas não publicam uma
execução parcial; uma execução existente nunca é sobrescrita. Para refazer uma
previsão com outra configuração ou versão, use outro `--output-dir`.

## Distribuição: avaliação

A separação entre modelo, aquisição e entrypoint é adequada. Processamento e
exportação também merecem módulos próprios: são contratos científicos e espaciais
que precisam ser verificados independentemente do transporte e dos pesos.

`01_ARAFNO` pode continuar como catálogo humano de modelos. O nome importável é
`arafno`, dentro de `src/`, e o wheel contém o runtime privado e o código do modelo.
Nenhuma outra pasta de modelo é necessária. Para
distribuir uma versão, entregue **wheel + bundle de artefatos versionado +
configuração**, mantendo pesos e estatísticas juntos e verificando seu manifesto.
O SHA-256 detecta alterações, mas não substitui a proveniência do bundle.

Os 93 MB de pesos devem ir para releases, armazenamento de objetos ou Git LFS;
estão ignorados no Git comum. Os `.npy` e o manifesto podem ser versionados.
Antes de uma distribuição pública, registre licença e origem dos pesos e as
condições aplicáveis de redistribuição dos dados. Nenhuma licença foi presumida.
Os originais em `reference/` servem à auditoria e não são entrypoints mantidos.

Um GeoTIFF por emissão e antecedência facilita SIG e consumo independente, sem
misturar previsões que têm a mesma validade e emissões diferentes. Para servir
recortes remotos em grande escala, uma evolução é converter para COG e catalogar
com STAC; estes arquivos são GeoTIFFs tiled comuns, não COGs certificados. Para
verificação científica de grandes séries, Zarr continua complementar, com eixos
de emissão e antecedência separados. Cache bruto, artefatos do modelo e produtos
derivados têm ciclos de vida distintos e não devem compartilhar uma pasta plana.

## Verificação

```bash
uv run --locked pytest
uv run --locked ruff check src tests
uv build
```

Os testes usam dados sintéticos identificáveis e modelos pequenos para verificar
tempo, orientação, normalização, aquisição simulada, carregamento estrito e
GeoTIFF. Isso testa o software e não substitui validação meteorológica com NOW.

Veja [VALIDATION.md](VALIDATION.md) para os resultados obtidos nesta entrega,
incluindo a execução dos pesos reais e a localização dos GeoTIFFs sintéticos.
