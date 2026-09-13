# DeepSeek V4.1-Flash — stato di questo branch: FERMO, per scelta

Questo branch contiene il nostro port di V4.1 sull'engine **amalgamato** (`c/deepseek_v41.c`,
~19.000 righe ereditate da V4 con le unità V4.1 innestate). **Non va completato.**

## Perché è fermo

Il mantainer ha pubblicato `feat/deepseek-v41-flash` (PR JustVugg/colibri#1453, 20 commit,
39 file, +7465): un engine **scritto per V4.1 da zero** — 3.522 righe, AVX2, blocchi di
posizioni (`mv8_rows`), MoE esperto-major, attenzione sparsa con `sparse_attn.h`, DSpark,
torre visiva, tool calling DSML, gateway e CI. Raggiungere la parità da qui costerebbe 4-6
sessioni di lavoro **verso un'implementazione peggiore**, e nessuno esegue questo engine:
il fork non è unito e non lo sarà. Quindi: congelato.

## Cosa c'è dentro, e a cosa serve ancora

Non è codice morto: è **lo strumento di verifica**, e su questa macchina è ciò che ha
permesso di trovare cose che dall'interno del loro motore non si vedono.

| file | a cosa serve |
|---|---|
| `c/tools/deepseek_v41_reference.py` | l'oracle che importa **il `model.py` del vendor intatto** dietro uno shim del modulo `kernel` (6 operatori foglia). Il loro oracle è una *riscrittura*: può condividere un fraintendimento con l'engine che valida. Questo no, per costruzione. |
| `c/tests/v41_ops_probe.c` + `c/tools/check_deepseek_v41_ops.py` | il pin degli operatori C contro il vendor, **con controlli negativi** (nibble invertiti → 9.0, blocco scala spostato → 2.6, fold ignorato → 189.0): un test di cui si sa che *può* fallire. |
| `c/tools/deepseek_v41_layout.py` | mappa nome→forma→quantizzazione, tenuta contro gli header del checkpoint (96.085 tensori, 0 non classificati) e contro ciò che il loader chiede. |
| `c/tools/make_deepseek_v41_tiny.py` + `c/deepseek_v41_tiny/` | fixture costruita dal modello del vendor **per falsificare** le regole di forma lette dal solo checkpoint (ne ha falsificate 3), con token attesi derivati dallo stesso modello. |

## Cosa ha prodotto (in locale, nulla è stato comunicato al mantainer)

- La quantizzazione delle **attivazioni** che il vendor fa in ogni `linear()` (`act_quant`,
  blocchi 32) e che engine+oracle del mantainer non fanno: **1,4-3,3%** di scarto per
  proiezione sui pesi fp8 veri, **2 token su 8** ribaltati sulla loro stessa fixture.
- La **fragilità del loro gate** token-per-token: su 5 seed della loro fixture, 4 passano e
  il *loro* seed di default sbaglia un token; un altro seed azzecca i token e sbaglia la
  testa DSpark con una confidenza a 4× di distanza.
- La ricetta delle tabelle engram verificata contro `engram.py` del vendor (primi, offset,
  moltiplicatori derivati dal vocabolario **compresso**), col guard che sul checkpoint vero
  è scattato esattamente sul numero atteso: **99.092** classi.

## Se un giorno serve

Serve a una cosa sola: **verificare un motore dall'esterno**, con il codice del vendor come
verità. Non a eseguire V4.1 — per quello si usa il loro engine (`start-v41.sh` sul disco 5).
