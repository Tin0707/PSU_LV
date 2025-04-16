def analiza_sms_poruka():
    ime_datoteke = "C:/Users/Tin/Desktop/SMSSpamCollection.txt"
    
    try:
        with open(ime_datoteke, 'r') as f:
            ham_count = 0
            spam_count = 0
            ham_word_count = 0
            spam_word_count = 0
            spam_ending_with_exclamation = 0
            
            for line in f:
                line = line.strip()
                label, message = line.split("\t", 1)
                words = message.split()
                
                if label == "ham":
                    ham_count += 1
                    ham_word_count += len(words)
                elif label == "spam":
                    spam_count += 1
                    spam_word_count += len(words)
                    if message.endswith('!'):
                        spam_ending_with_exclamation += 1
            
            if ham_count > 0:
                avg_ham_words = ham_word_count / ham_count
                print(f"Prosječan broj riječi u 'ham' porukama: {avg_ham_words:.2f}")
            else:
                print("Nema 'ham' poruka.")
            
            if spam_count > 0:
                avg_spam_words = spam_word_count / spam_count
                print(f"Prosječan broj riječi u 'spam' porukama: {avg_spam_words:.2f}")
            else:
                print("Nema 'spam' poruka.")
            
            print(f"Broj 'spam' poruka koje završavaju uskličnikom: {spam_ending_with_exclamation}")
    
    except FileNotFoundError:
        print(f"Datoteka {ime_datoteke} nije pronađena.")

analiza_sms_poruka()
