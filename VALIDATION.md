# Validação realizada

Data: 19 de outubro de 2026. Ambiente de execução: Linux, Python 3.12, CPU.

## Resultados

- `pytest -q`: **13 testes aprovados**. Cobertura de janela horária e virada do dia,
  nomes NOW com intervalo, timezone, recorte com marcadores assimétricos,
  orientação, normalização, faltantes, arquivo inválido, pesos incompatíveis,
  integridade dos artefatos, coordenadas GeoTIFF, inferência integrada, publicação
  sem sobrescrita, falha de escrita e cache FTP simulado.
- `ruff check src tests`: aprovado.
- `uv build`: wheel e pacote de fontes gerados. Ambos excluem os pesos e os
  scripts de referência; estes são distribuídos separadamente.
- Checkpoint real carregado estritamente. A comparação com as classes originais
  de `reference/model.py`, usando os mesmos pesos e uma entrada nula normalizada,
  apresentou diferença absoluta máxima **0,0** em uma previsão de uma hora.
- Pipeline completo executado com o checkpoint real e seis binários GSMaP
  **sintéticos**. Foram gerados seis GeoTIFFs de 880 × 970 pixels. Todos foram
  reabertos e tiveram CRS, bordas, horários, unidade, valores finitos não negativos
  e SHA-256 conferidos contra o manifesto.

Os produtos sintéticos estão em:

```text
outputs/validation-synthetic/arafno/gsmap_now/2000/01/01/0500Z/
```

Esses arquivos demonstram o funcionamento técnico; **não são previsões
meteorológicas reais**. Os binários sintéticos de entrada foram temporários e
removidos ao final do ensaio. O manifesto conserva seus nomes e hashes.

## O que ainda não foi validado

- Download no servidor JAXA real: as variáveis de credenciais não estavam
  configuradas. Listagem, cache e recuperação de arquivo corrompido foram testados
  com um transporte simulado.
- Ordem das latitudes e limites efetivamente utilizados no treinamento:
  não constam do checkpoint. A configuração usa a orientação norte→sul do
  script operacional e o domínio dos scripts fornecidos.
- Habilidade meteorológica com NOW: os scripts de inferência originais usavam
  NRT. O pipeline usa NOW por solicitação, preservando as estatísticas fornecidas.
- GPU/CUDA e serviço operacional agendado: não foram executados neste ensaio.

As instruções de execução e a avaliação da organização para distribuição estão
no [README](README.md).

## Execução com arquivos NOW reais

Após a execução FTP feita pelo usuário, os seis arquivos de 19/10/2026, das 15h
às 20h UTC, foram inspecionados no cache. Cada quadro apresentou entre 0,29873%
e 0,38824% de pixels faltantes, todos com valor −99. A política `error` interrompia
corretamente a execução; a configuração operacional foi ajustada para
`normalized_zero`, mantendo o limite de 5% por quadro.

O pipeline foi reexecutado em CPU, em modo local sobre esse cache, com emissão
às 20h UTC. Os seis GeoTIFFs foram publicados em
`outputs/arafno/gsmap_now/2026/10/19/2000Z/`, com validade das 21h de 19/10 às
02h de 20/10. A união dos pixels faltantes na janela corresponde a 0,42549% do
domínio e permanece como NoData nas saídas. A orientação do treinamento e a
habilidade meteorológica em NOW continuam pendentes de validação.

## Ajuste posterior: negativos convertidos em zero

Por solicitação do usuário, valores finitos negativos agora são convertidos em
zero mm/h antes da normalização, independentemente de `missing_policy`. NaN e
infinitos continuam seguindo a política de faltantes. Assim, a descrição de −99
como NoData na execução acima registra o comportamento anterior.

Após o ajuste, **18 testes passaram**, incluindo negativos sob as três políticas
de faltantes e NaN/±infinito. Ruff também passou. O processamento dos seis arquivos
reais do cache apresentou zero pixels faltantes e nenhuma máscara de saída.
Os GeoTIFFs da execução anterior não foram regenerados; para reprocessar a mesma
emissão, use um novo `--output-dir`.

## Remoção de parâmetros redundantes

As opções operacionais do JSON não têm mais valores padrão duplicados nas
dataclasses. A resolução nativa e a grade global pertencem a `formats.py`, com
dimensões e tamanho binário derivados. CRS e unidade usam a mesma fonte no
exportador e no manifesto. A arquitetura é reconstruída do checkpoint, sem
hiperparâmetros padrão no construtor, e o produto de referência vem do manifesto
de artefatos.

**20 testes aprovados** e Ruff aprovado após a alteração. A conferência dos pesos
reais e das seis observações do cache preservou a entrada `(6, 880, 970)`, sem
pixels mascarados. Nenhuma previsão existente foi alterada.

## Agendamento e retentativas — 20/10/2026

- **29 testes aprovados**, incluindo polling sem dados, emissão já concluída,
  proteção contra concorrência, fixação do timestamp descoberto, nova consulta
  após falha, dados antigos/futuros e integridade das saídas já publicadas.
- Transferências simuladas cobriram erro de rede, gzip com grade incompleta,
  recuperação com espera crescente e esgotamento das tentativas sem publicar
  um arquivo inválido no cache.
- Ruff aprovado. As unidades geradas passaram por `systemd-analyze verify`;
  a expressão de calendário foi validada com `systemd-analyze calendar`.
- Os parâmetros de transporte têm uma única fonte na seção `ftp` do JSON;
  a frequência tem uma única fonte no arquivo `.timer`.

Não houve ativação do serviço: o acesso ao bus de usuário nesta sessão retornou
`Operation not permitted`. Não foi executado polling contínuo contra o FTP real.
Instalação, credenciais e ativação estão descritas em [deploy/README.md](deploy/README.md).

## Runtime compartilhado — 21/10/2026

Aquisição, processamento, GeoTIFF, publicação e agendamento foram extraídos para
`shared/nowcasting_runtime`, utilizado também pelo novo pipeline Xception. Os
imports e comandos ARAFNO permanecem disponíveis por adaptadores locais.
**Os 29 testes ARAFNO passaram novamente** no ambiente próprio após a extração.
As unidades systemd foram regeneradas e verificadas, sem ativação. O pacote e o
lock foram atualizados para instalar o runtime local; execute `uv sync --locked`
ao atualizar outra instalação.

## Distribuição independente — 21/10/2026

- O runtime de aquisição, processamento, publicação e agendamento está em
  `src/arafno/_runtime/` e é empacotado no mesmo wheel do modelo.
- `uv lock --offline` e `uv build --offline` passaram. O wheel contém os 11
  módulos do runtime privado e não declara dependência de `nowcasting-runtime`
  nem de um caminho `../shared`. A importação do runtime a partir de um wheel
  extraído em `/tmp`, fora do checkout, também passou.
- 29 testes passaram com `uv run --offline --locked --group dev pytest -q`.
- A independência exige manter correções do runtime em cada modelo que precisar
  delas; mudanças em uma pasta não atualizam automaticamente as outras.

## Retenção temporária dos downloads — 21/10/2026

- O cache FTP padrão passou para o diretório temporário do sistema, isolado por
  usuário e modelo. `--cache-ttl-hours` define a retenção (24 horas por padrão).
- A limpeza remove somente downloads vencidos em namespaces gerenciados; arquivos
  locais e outros diretórios não são tocados. Ela roda também no polling que
  encontra uma emissão já concluída.
- **31 testes aprovados** com `uv run --offline --locked --group dev pytest -q`;
  Ruff e `uv lock --offline --check` aprovados. O calendário de cinco minutos
  passou em `systemd-analyze calendar`.
- A reconstrução do wheel nesta sessão ficou pendente: o cache offline não
  contém `hatchling`, exigido pelo build. O build anterior à alteração passou;
  a alteração atual foi validada por testes no código-fonte.
