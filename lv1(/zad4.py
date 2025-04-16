def izracunaj_pouzdanost():
    ime_datoteke = input("Unesite ime datoteke: ")
    
    try:
        with open(ime_datoteke, 'r') as f:
            total = 0
            count = 0
            
            for linija in f:
                if linija.startswith("X-DSPAM-Confidence:"):
                    try:
                        broj = float(linija.split(":")[1].strip())
                        total += broj
                        count += 1
                    except ValueError:
                        continue
            
            if count > 0:
                print(f"Prosječna X-DSPAM-Confidence: {total / count}")
            else:
                print("Nema podataka o pouzdanosti u datoteci.")
    
    except FileNotFoundError:
        print(f"Datoteka {ime_datoteke} nije pronađena.")

izracunaj_pouzdanost()
