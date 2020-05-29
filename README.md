# CLARANS-Clustering-Algorithm

### Środowisko

Do stworzenia środowiska potrzebne są:
- `python3.6`
- `python3-venv` (Linux:`sudo apt install python3-venv`)

### Tworzenie środowiska
- `python3 -m venv venv` (tworzy środowisko `venv`)
- `source venv/bin/activate` (aktywacja środowiska)
- `pip3 install --upgrade pip` (upgrade pip'a)
- `pip3 install -r requirements.txt` (instalacja dodatkowych bibliotek)

#### Instalacja basemap
- `sudo apt-get install libgeos-dev`
- `sudo pip3 install -U git+https://github.com/matplotlib/basemap.git`

### Testowanie
- wykonaj komendę `pytest` w katalogu projektu lub `make test`