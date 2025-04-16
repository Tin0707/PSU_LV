def broj_rijeci_koje_se_jednom_pojavljuju():
    ime_datoteke = "song.txt"
    
    try:
        with open(ime_datoteke, 'r') as f:
            rijeci = f.read().split()
            rjecnik = {}
            
            for rijec in rijeci:
                rijec = rijec.lower().strip('.,!?":;()[]{}')  # Uklanjanje interpunkcija i pretvaranje u mala slova
                rjecnik[rijec] = rjecnik.get(rijec, 0) + 1
            
            rijeci_koje_se_jednom_pojavljuju = [rijec for rijec, broj in rjecnik.items() if broj == 1]
            print(f"Broj riječi koje se pojavljuju samo jednom: {len(rijeci_koje_se_jednom_pojavljuju)}")
            print("Riječi koje se pojavljuju samo jednom:", rijeci_koje_se_jednom_pojavljuju)
    
    except FileNotFoundError:
        print(f"Datoteka {ime_datoteke} nije pronađena.")

broj_rijeci_koje_se_jednom_pojavljuju()
