import pandas as pd

def bayes_prediction(data, ciel, vent):
    omega = len(data)
    total_yes = len(data[data["Jouer au tennis"] == "Oui"])
    total_no = len(data[data["Jouer au tennis"] == "Non"])

    p_yes = total_yes / omega
    p_no = total_no / omega

    # probabilité de jouer
    p_c_yes = len(data[(data["Ciel"] == ciel) & (data["Jouer au tennis"] == "Oui")]) / total_yes
    p_v_yes = len(data[(data["Vent"] == vent) & (data["Jouer au tennis"] == "Oui")]) / total_yes
    p_yes_cv = p_yes * p_c_yes * p_v_yes

    # probabilité de ne pas jouer
    p_c_no = len(data[(data["Ciel"] == ciel) & (data["Jouer au tennis"] == "Non")]) / total_no
    p_v_no = len(data[(data["Vent"] == vent) & (data["Jouer au tennis"] == "Non")]) / total_no
    p_no_cv = p_no * p_c_no * p_v_no

    return p_yes_cv > p_no_cv

"""
def run_bayes():

    data = pd.read_csv("file_plus_1.csv")

    print("=== Prédiction Bayes Naïve : Jouer au Tennis ===")

    # Saisie utilisateur
    day = int(input("\nEntrer le jour (1-30) : "))
    sky = int(input("Entrer le ciel (Ensoleillé: 1, Couvert: 2, Pluie: 3) : "))
    vent = int(input("Entrer le vent (Fort: 1, Faible: 2) : "))

    # Conversion en texte
    ciel_map = {1: "Ensoleillé", 2: "Couvert", 3: "Pluie"}
    vent_map = {1: "Fort", 2: "Faible"}

    ciel = ciel_map.get(sky)
    vent = vent_map.get(vent)

    # Calcul Bayes
    prediction = bayes_prediction(data, ciel, vent)

    if prediction:
        print(f"Oui, le joueur va jouer au tennis au jour {day}")
    else:
        print(f"Non, le joueur ne va pas jouer au tennis au jour {day}")
"""

import pandas as pd

def run_bayes():

    # Dataset défini directement dans le code
    dataset = {
        "Ciel": ["Ensoleillé","Ensoleillé","Couvert","Pluie","Pluie","Pluie",
                 "Couvert","Ensoleillé","Ensoleillé","Pluie","Ensoleillé"],
        "Vent": ["Faible","Fort","Faible","Faible","Faible","Fort",
                 "Fort","Faible","Faible","Faible","Fort"],
        "Jouer au tennis": ["Non","Non","Oui","Oui","Oui","Non",
                  "Oui","Non","Oui","Oui","Non"]
    }

    data = pd.DataFrame(dataset)

    print("=== Prédiction Bayes Naïve : Jouer au Tennis ===")

    # Saisie utilisateur
    day = int(input("\nEntrer le jour (1-30) : "))
    sky = int(input("Entrer le ciel (Ensoleillé: 1, Couvert: 2, Pluie: 3) : "))
    vent = int(input("Entrer le vent (Fort: 1, Faible: 2) : "))

    # Conversion en texte
    ciel_map = {1: "Ensoleillé", 2: "Couvert", 3: "Pluie"}
    vent_map = {1: "Fort", 2: "Faible"}

    ciel = ciel_map.get(sky)
    vent_user = vent_map.get(vent)

    # Calcul Bayes
    prediction = bayes_prediction(data, ciel, vent_user)

    if prediction:
        print(f"Oui, le joueur va jouer au tennis au jour {day}")
    else:
        print(f"Non, le joueur ne va pas jouer au tennis au jour {day}")