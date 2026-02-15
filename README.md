## PV Battery Planner

Een hulpmiddel om het laden van een thuisbatterij te plannen op basis van:
- weersvoorspellingen (Open‑Meteo / ECMWF),
- PV‑systeemconfiguratie (opbrengstschatting met `pvlib`),
- je verbruiksprofiel (geschaald op gisteren),
- en je elektriciteitstarief (daluren/piekuren).

De applicatie geeft concrete instellingen voor FusionSolar:
- **Allowed AC charge power (kW)**  
- **AC charge cutoff SOC (%)**

Daarnaast kun je de volledige dagbalans (PV, verbruik, batterij‑SOC, netimport/export) grafisch inspecteren.

---

### 1. Features

- **Web UI (Streamlit)**
  - Hoofdinterface in `app.py` (`PV Battery Charging Planner`)
  - Experimentele v2‑interface in `app_v2.py`
  - Interactieve grafieken met Plotly
  - Historiek van uitgevoerde plannen in `run_history_log.json`
- **PV‑modellering**
  - Weerdata via Open‑Meteo (`fetch_tomorrow_weather`)
  - PV‑opbrengstberekening met `pvlib` (oost- en zuidveld, clipping aan omvormerlimiet)
  - Load‑profiel op uurbasis geschaald naar gisterenverbruik
- **Tariefmodel**
  - Daluren per weekdag configureerbaar in `config.json` of via UI
  - Piekuren = complement van daluren
- **Batterijplanning**
  - Berekent minimale SOC om dure uren te overbruggen
  - Houdt rekening met verwacht PV‑overschot overdag (headroom)
  - Plant nachtelijk AC‑laden in daluren met vermogens- en SOC‑limieten

Meer detail staat in:
- `docs/gebruikershandleiding.md`
- `docs/technische_documentatie.md`

---

### 2. Installatie

#### 2.1. Vereisten

- Python 3.10+ (aanbevolen)
- Internettoegang (voor Open‑Meteo API)

#### 2.2. Virtuele omgeving (optioneel maar aangeraden)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

#### 2.3. Dependencies

In de projectroot:

```bash
pip install -r requirements.txt
```

Als `requirements.txt` ontbreekt, installeer minstens:

```bash
pip install pandas requests pvlib streamlit plotly
```

---

### 3. Applicatie starten

#### 3.1. Web UI v1 (hoofdapp)

```bash
streamlit run app.py
```

Open daarna de URL die Streamlit toont (meestal `http://localhost:8501`).

#### 3.2. Web UI v2 (experimentele UI)

```bash
streamlit run app_v2.py
```

Deze versie heeft een andere layout (meer tabbladen), maar gebruikt dezelfde planner‑logica.

#### 3.3. CLI‑modus (advanced)

Je kunt de planner ook via de command line draaien:

```bash
python planner_core.py
```

Je wordt dan interactief om:
- **Battery SOC at 22:00 (%)**
- **Total consumption yesterday (kWh)**
gevraagd en krijgt een tekstuele samenvatting + FusionSolar‑instellingen in de terminal.

---

### 4. Configuratie

De basisconfiguratie staat in `config.json`. Belangrijke blokken:

- **`location`**: adres, breedte‑/lengtegraad, tijdzone
- **`pv`**: paneelvermogen, aantal panelen oost/zuid, tilt, azimuth, performance ratio, omvormerlimiet
- **`battery`**: capaciteit, min/max SOC, laad/ontlaad‑kW, AC‑laadlimiet
- **`load_profile`**: 24 waarden (relatief verbruiksprofiel per uur)
- **`tariff.offpeak_windows_by_dow`**: daluren per weekdag

Je kunt deze waarden:
- aanpassen via de Streamlit UI (`Settings (saved)`), of
- direct in `config.json` bewerken (app herstart vereist).

Details over elk veld staan in `docs/technische_documentatie.md`.

---

### 5. Workflow in het kort

1. Start de Streamlit‑app (`app.py` of `app_v2.py`).
2. Stel scenario‑inputs in:
   - SOC om 22:00 (%)
   - Totale consumptie gisteren (kWh)
   - Optioneel: safety buffer SOC (%) en max AC‑laadvermogen (kW)
3. Pas indien nodig PV/batterij/tarief‑configuratie aan onder **Settings (saved)**.
4. Klik **Run forecast**.
5. Lees de voorgestelde:
   - **Allowed AC charge power (kW)**
   - **AC charge cutoff SOC (%)**
   en configureer deze in FusionSolar.
6. Bekijk de grafieken (PV vs load, surplus/deficit, grid import/export) en de tabellen voor detailanalyse.

---

### 6. Documentatie

- **Gebruikershandleiding**: zie `docs/gebruikershandleiding.md`
- **Technische documentatie (ontwikkelaars)**: zie `docs/technische_documentatie.md`

# PV Battery Planner

## Supported Python versions

- Recommended: **Python 3.12 x64** for best wheel availability.
- Supported: **Python 3.11–3.13 x64**.
- Unsupported for this project setup: **32-bit Python** and free-threaded Python builds.

## Windows quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `pandas` fails because pip is building from source, install **Python 3.12.x (64-bit)**, delete and recreate `.venv`, then rerun the commands above.

## Windows setup (one command)

### Prerequisite

Install **64-bit CPython** (recommended: Python 3.12 x64):
https://www.python.org/downloads/windows/

### Install project dependencies

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

This script runs a Python preflight check, creates `.venv` if needed, upgrades packaging tools, and installs dependencies with wheel-only mode (`--only-binary=:all:`) to avoid source builds.

### Run the Streamlit GUI

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_gui.ps1
```

The run script always starts Streamlit as:

```powershell
python -m streamlit run app.py
```

which avoids PATH issues where `streamlit` is not recognized.


## How to run on Windows (manual commands)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not recognized, use:

```powershell
python -m streamlit run app.py
```

## Troubleshooting (Windows)

- If `pandas` tries to build from source, your Python is likely 32-bit or an unsupported build.
- Confirm architecture:

  ```powershell
  python -c "import platform; print(platform.architecture())"
  ```

- Confirm pip in the active interpreter:

  ```powershell
  python -m pip --version
  ```

- If preflight fails, install Python 3.12 x64, recreate `.venv`, and rerun:

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
  ```


## Units and columns

- Weather inputs are normalized to **Europe/Brussels** local hourly timestamps (00:00-23:00).
- Wind speed is handled in **m/s**.
- Temperature is handled in **°C**.

PV forecast columns:

- `pv_east_kwh`, `pv_south_kwh`: per-array **AC** energy (unclipped).
- `pv_total_unclipped_kwh`: total **AC** energy before inverter clipping.
- `pv_total_kwh`: total **AC** energy after inverter clipping.
- `pv_clipped_kwh`: clipped energy (`pv_total_unclipped_kwh - pv_total_kwh`).
- Legacy aliases are kept for backward compatibility: `pv_dc_available_*` maps to total unclipped AC, and `pv_ac_limited_*` maps to clipped AC output.

## Persistent settings

- The planner now reads persistent settings from `config.json` stored in the project root (same folder as `planner_core.py`).
- In Streamlit, open **Settings (saved)** in the left panel, update values, and click **Save settings**.
- Settings are used by both the Streamlit app and CLI on the next run automatically.
- If `config.json` is missing, built-in defaults are used.
- Delete `config.json` to revert to default settings.

## Run terminal script

```bash
python pv_battery_planner.py
```

## Smoke test (offline)

Use this to verify imports and core planner logic without any API/network calls:

```bash
python scripts/smoke_test.py
```

## Full offline health check

Use this to run syntax checks, bytecode compilation, and the smoke test in one command:

```bash
python scripts/full_check.py
```
