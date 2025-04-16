def unos_brojeva():
    brojevi = []
    
    while True:
        unos = input("Unesite broj ili 'Done' za završetak: ")
        
        if unos.lower() == 'done':
            break
        
        try:
            broj = float(unos)
            brojevi.append(broj)
        except ValueError:
            print("To nije broj. Pokušajte ponovo.")
    
    if len(brojevi) > 0:
        print(f"Broj unesenih brojeva: {len(brojevi)}")
        print(f"Srednja vrijednost: {sum(brojevi) / len(brojevi)}")
        print(f"Minimalna vrijednost: {min(brojevi)}")
        print(f"Maksimalna vrijednost: {max(brojevi)}")
        brojevi.sort()
        print(f"Sortirani brojevi: {brojevi}")
    else:
        print("Niste unijeli nijedan broj.")

unos_brojeva()
