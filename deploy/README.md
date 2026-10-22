# Agendamento com systemd

O agendamento é feito por um **timer do systemd**, separado do processamento.
O serviço executa uma consulta com `--scheduled` e termina. O timer consulta
novamente a cada cinco minutos, independentemente de ter encontrado dados novos.
Não é necessário manter um terminal aberto ou um loop Python em segundo plano.

O contrato deste modelo é uma sequência **horária**: somente nomes `HH00` são
selecionados. O NOW também possui atualizações sub-horárias, mas `HH30` não é
entrada deste pipeline. Consultar a cada cinco minutos reduz a espera depois da
publicação de uma nova emissão horária, sem presumir um minuto exato de chegada.
A detecção acontece na primeira consulta bem-sucedida após a publicação; não há
notificação instantânea enviada pelo provedor.

## Funcionamento de cada consulta

1. Obtém um lock exclusivo para o produto e o diretório de saída. Outra consulta
   agendada em andamento resulta em `BUSY`, sem iniciar uma segunda execução.
2. Lista o FTP e identifica o horário mais recente. Não carrega os pesos nem
   baixa arquivos se esse horário já tem uma saída completa e íntegra.
3. Exige dados atuais (`--max-age-hours`, definido na CLI), fixa o horário da
   consulta e tenta obter a janela completa exigida pelo checkpoint.
4. Se faltam horários, registra `WAITING` e termina normalmente. O timer consulta
   novamente depois. Falhas de rede ou de download são tentadas conforme `ftp`
   em `config/latam.json`: três tentativas por operação, timeout de 60 segundos,
   espera inicial de 30 segundos e crescimento exponencial limitado a 120 segundos.
   Com três tentativas, as esperas são de 30 e 60 segundos.
5. Um gzip truncado, CRC inválido ou tamanho binário errado também dispara uma
   nova tentativa de transferência. O cache só recebe arquivos validados.
6. Se todas as tentativas falham, o processo retorna erro e o journal registra a
   falha. O timer tenta outra vez no próximo disparo. Não existe um segundo
   agendador de retentativas concorrendo com o timer.
7. Gera os GeoTIFFs e publica o manifesto completo. As próximas consultas mostram
   `UP_TO_DATE` para essa emissão; os arquivos não são reprocessados nem sobrescritos.

O sistema prioriza a **emissão mais recente disponível**. Após uma interrupção
longa, não recupera automaticamente todas as emissões intermediárias. O timer
persistente retoma a consulta ao iniciar a sessão/servidor, mas não cria uma fila
de previsões históricas. Para isso, use reprocessamento com `--run-time` fora do
modo agendado. Revisões dos dados sob o mesmo timestamp não geram nova previsão.

## Retenção dos dados de entrada

Os downloads FTP ficam no diretório temporário do sistema, isolados por
usuário e modelo (`nowcasting-<UID>/<modelo>`). O prazo padrão é 24 horas
desde o download. Cada disparo do serviço remove os arquivos expirados,
inclusive quando a previsão já está pronta; não apaga entradas locais
fornecidas por `--input-dir`. A opção `--cache-ttl-hours` altera o prazo
e `--cache-dir` escolhe outro cache gerenciado. A limpeza também ocorre
na próxima execução FTP manual, mas não há processo residente para
apagar arquivos enquanto o timer estiver desativado. Dados antigos em
`data/raw/` não são removidos automaticamente por esta mudança.

## Preparação

Execute na pasta `01_ARAFNO`:

```bash
uv sync --locked
cp -n deploy/ftp.env.example .env
chmod 600 .env
${EDITOR:-vi} .env
```

Preencha usuário e senha no arquivo `.env`. O formato é `CHAVE=valor`, sem
`export`. O arquivo é ignorado pelo Git e lido pelo systemd; a CLI executada
manualmente continua recebendo variáveis do ambiente. Não use credenciais na
linha de comando do serviço. Senhas com espaços devem ser colocadas entre aspas
conforme o formato `EnvironmentFile` do systemd.

Gere as unidades com o caminho absoluto desta instalação:

```bash
.venv/bin/python deploy/render_units.py
systemd-analyze verify deploy/generated/arafno-now.service deploy/generated/arafno-now.timer
```

O gerador **somente escreve em `deploy/generated/`**. Ele não modifica serviços
do computador nem ativa o agendamento. Após mover o projeto, gere e instale as
unidades novamente. A instalação deve estar em um filesystem acessível na
máquina que vai executar o serviço, com suporte a locks `flock`.

## Instalação e ativação no seu usuário

```bash
mkdir -p "$HOME/.config/systemd/user"
cp deploy/generated/arafno-now.service deploy/generated/arafno-now.timer "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now arafno-now.timer
systemctl --user start arafno-now.service
```

O último comando executa uma consulta imediatamente. O timer mantém as próximas
consultas. Os argumentos de execução usam os padrões da mesma CLI do pipeline,
sem duplicar horizonte, dispositivo ou diretórios na unidade. Ela usa a `.venv`
já instalada, sem chamar `uv sync` ou instalar pacotes a cada consulta.

Para continuar executando após logout e iniciar no boot, o gerenciador de serviços
do usuário precisa de **linger**. Verifique com:

```bash
loginctl show-user "$USER" -p Linger
```

Se retornar `Linger=no`, peça ao administrador para habilitar:

```bash
sudo loginctl enable-linger "$USER"
```

Se `systemctl --user` não puder conectar ao bus, execute esses comandos em uma
sessão normal do usuário na máquina de operação. Em um cluster, o administrador
deve indicar um nó de serviço permitido; não presuma que processos permanentes
são permitidos no nó de login. Nesta sessão de desenvolvimento o acesso ao bus
foi negado, portanto **as unidades foram preparadas, mas não ativadas**.

## Acompanhamento e ajustes

```bash
systemctl --user list-timers arafno-now.timer
systemctl --user status arafno-now.service
journalctl --user -u arafno-now.service -f
```

`inactive (dead)` para um serviço `oneshot` que concluiu com sucesso é normal;
o timer deve permanecer ativo. Os resultados usam a mesma árvore
`outputs/arafno/gsmap_now/AAAA/MM/DD/HHMMZ/`.

- Frequência: altere somente `OnCalendar` em `deploy/arafno-now.timer`. O valor
  entregue é `*-*-* *:0/5:00 UTC`. Depois gere, copie as unidades novamente e rode
  `systemctl --user daemon-reload` e `systemctl --user restart arafno-now.timer`.
- Retentativas: altere somente a seção `ftp` em `config/latam.json`. O arquivo é
  relido a cada consulta. Configurações antigas precisam incluir essa seção.
- Horizonte/dispositivo/pastas: acrescente os argumentos necessários ao comando
  gerado pelo script e regenere a unidade. Para outra versão dos pesos ou outra
  configuração científica, utilize outro `--output-dir`, pois emissões concluídas
  no mesmo diretório são preservadas.
- Prazo de uma execução: `TimeoutStartSec=20min` limita uma consulta travada.
  O timer não inicia novamente o mesmo serviço enquanto ele ainda está rodando.
- Parar novas consultas: `systemctl --user disable --now arafno-now.timer`.
  Para interromper também uma execução em andamento, use
  `systemctl --user stop arafno-now.service`.

O lock protege invocações **agendadas** com o mesmo diretório e produto. Para uma
execução manual paralela, escolha outro diretório de saída. Uma execução
interrompida não publica uma emissão parcial. Interrupção forçada pode deixar
pastas temporárias ocultas, que não contam como previsões completas.

Referências: [timers do systemd](https://www.freedesktop.org/software/systemd/man/252/systemd.timer.html),
[formato GSMaP NOW](https://sharaku.eorc.jaxa.jp/GSMaP/document/new/DataFormatDescription_NOW.pdf).
