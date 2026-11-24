# Data provenance

## Hourly grid carbon intensity — `data/raw/*.csv`

- **Original source:** Electricity Maps historical datasets (hourly, 2021–2024), columns
  `Datetime (UTC)`, `Carbon intensity gCO₂eq/kWh (direct)`, `Carbon intensity gCO₂eq/kWh (Life cycle)`,
  CFE %, RE %, data source, estimation flags. Licence: Open Database License (ODbL).
  Cite as: Electricity Maps, "<Zone> <Year> hourly carbon intensity data", https://www.electricitymaps.com
- **Retrieved from:** the CarbonX repository, `data/forecasting-data/<ZONE>.csv`,
  https://github.com/codecexp/CarbonX, commit `542536f43ba2ad97e2a6f779db98fdeea3ac1c69`
  (2 Oct 2025), Apache-2.0. Files are copied unchanged.
- **Zones (23):** FR, FI, SE, DE, GB, PL, ES, IN-WE, JP-TK, AU-NSW, AU-VIC, US-CAL-CISO,
  US-TEX-ERCO, US-MIDA-PJM, US-MIDW-MISO, US-NW-BPAT, US-NW-PACW, US-NW-PACE, US-CAR-DUK,
  US-SW-PNM, US-CENT-SWPP, US-TEN-TVA, US-SE-SOCO.
- **Cleaning:** reindexed to a common hourly UTC axis (35,064 h); missing CI values filled by
  linear interpolation in time (US-CAR-DUK 65 h, US-NW-PACW 467 h; all others 0 h).
- **Excluded from the hourly-vs-annual analysis:** IN-WE and JP-TK (100 % of hourly values are
  modelled estimates by Electricity Maps, so their hourly variation is not measured).
- **Signal used:** direct (operational, location-based) intensity. Life-cycle intensity is
  1.10–1.45x higher for these zones (results/extras.json); reported as a scope sensitivity.

## Training-run disclosures — `src/config.py`

| Run | Source | Numbers taken |
|---|---|---|
| BLOOM-176B | Luccioni, Viguier, Ligozat, JMLR 24(253), 2023; BLOOM model card (Hugging Face) | 1,082,990 GPU-h; 384 A100-80GB avg; 118 d 5 h 41 m; start 11 Mar 2022 11:42 PST; Jean Zay, France; PUE 1.2; CI 57 g/kWh; 24.69 t dynamic, 14.6 t idle, 11.2 t embodied, 50.5 t total; 150 kg/GPU, 2,500 kg/node, 6-yr life, 85 % use |
| LLaMA-65B | Touvron et al., arXiv:2302.13971, Table 15 & §2.4 | 1,022,362 GPU-h; 400 W; PUE 1.1; 0.385 kg/kWh; 449 MWh; 173 t; 2048 A100-80GB, ~21 days |
| Llama 2 70B | Touvron et al., arXiv:2307.09288, Table 2; Llama 2 model card | 1,720,320 GPU-h; 400 W; 291.42 t; trained Jan–Jul 2023; RSC + production clusters |
| Llama 3.1 405B | Meta Llama 3.1 model card (github.com/meta-llama/llama-models) | 30.84M H100-80GB GPU-h; 700 W; 8,930 t location-based; 0 t market-based |
| OLMo-7B (A100) | Groeneveld et al., arXiv:2402.00838, Table 6 | 104 MWh IT energy, PUE 1.1 applied on top; 0.610 kg/kWh; 70 t (=104×1.1×0.610); 27 nodes × 8 A100-40GB; Australia |
| OLMo-7B (LUMI) | same | 135 MWh; reported 0 t (used only in extras) |

## Embodied-carbon prior sources
- Falk et al., arXiv:2509.00093 (2025): A100 cradle-to-gate 127.6 kg, cradle-to-grave 141 kg per GPU.
- NVIDIA HGX H100 PCF summary (WSP, ISO 14067): 1,312 kgCO₂e cradle-to-gate for the 8-GPU HGX system.
- Schneider et al. (Google), arXiv:2502.01671 (2025): 386–1,101 kgCO₂e embodied per TPU chip incl. host CPU and
  data-centre share; server LCAs vary ~40x (423–15,593 kg).
- Luccioni et al. 2023: 150 kg per A100 + 2,500 kg per server node.

## Meta campus list (plausible regions for Meta runs)
Campus locations from public data-centre directories (datacenters.com Meta listing, datacentermap.com).
The campus → balancing-authority mapping is the author's assumption, listed in `src/config.py`.
