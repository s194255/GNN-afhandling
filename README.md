# Grafbaserede neurale netværk til forudsigelse af molekylære egenskaber vha. repræsentationslæring

Dette projekt bruger molekylære grafneurale netværk til at prædiktere kemiske egenskaber ved hjælp af forskellige trænings- og fortræningsopgaver.

## Installation

Følg nedenstående trin for at installere alle nødvendige pakker.

1. Klon projektet fra GitHub:
    ```sh
    git clone https://github.com/s194255/GNN-afhandling.git
    cd GNN-afhandling
    ```

2. Installer nødvendige python-pakker:
    ```sh
    pip install -r requirements.txt
    ```

## Konfiguration

Inden du kører programmerne, skal du oprette en YAML-fil til at angive stien til, hvor datasættet skal gemmes.

1. Opret filen `config/data_roots.yaml`:
    ```sh
    echo "data_root: <sti_til_dér_hvor_datasættet_skal_gemmes>" > config/data_roots.yaml
    ```
   
   Erstat `<sti_til_dér_hvor_datasættet_skal_gemmes>` med den faktiske sti, hvor du ønsker at gemme datasættet.

## Kørsel af programmerne

Dette projekt indeholder to hovedprogrammer, som kan køres efter installationen:

1. **eksp2.py**: Dette script kører eftertræning (træning på efteropgaven).
    ```sh
    python src/eksp2.py
    ```

2. **fortræn.py**: Dette script kører fortræningsopgaven.
    ```sh
    python src/fortræn.py
    ```

## Yderligere oplysninger

For yderligere oplysninger henvises til projektets dokumentation og kildefiler.
