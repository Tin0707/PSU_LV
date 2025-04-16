def ocjena_kategorija():
    try:
        ocjena = float(input("Unesi ocjenu između 0.0 i 1.0: "))
        if ocjena >= 0.9:
            print("A")
        elif ocjena >= 0.8:
            print("B")
        elif ocjena >= 0.7:
            print("C")
        elif ocjena >= 0.6:
            print("D")
        elif ocjena < 0.6:
            print("F")
        else:
            print("Ocjena je izvan dozvoljenog intervala.")
    except ValueError:
        print("Greška! Unesite broj.")
        
ocjena_kategorija()
