# Teil 1: Daten verstehen
#### Wie viele Zeilen hat der Datensatz vor und nach dropna()?
vorher 891 \
nachher 714

#### Welche Bedeutung haben die Variablen:
        pclass = Klasse der Passagiere von 1 bis 3
        sex = Geschlecht der Passagiere
        age = Alter
        fare = der bezahlte Ticketpreis

#### Welche Zielvariable wird vorhergesagt?
ob diese Person überlebt hat

# Teil 2: Aufteilung der Daten

#### Wie viele Datenpunkte sind im:
Trainingsset (X_train.shape) = 499
Testset (X_test.shape) = 215

#### Welcher Prozentsatz wird für das Testset verwendet?
30%

# Teil 3: Ergebnisse interpretieren
Schau dir die Ausgabe von classification_report an.

#### Wie hoch ist die Accuracy?
77%

#### Wie hoch sind:
    Precision (Klasse 1 = überlebt) 0.8 tot, 0.72 überlebt
    Recall (Klasse 1) 0.81

#### Erkläre in eigenen Worten und in Bezug auf die Ergebnisse:

#### Was bedeutet der erreichte Wert bei Precision?
    80% aller überlebenden passagiere wurden erfolgreich klassifiziert
#### Was bedeutet der erreichte Wert bei Recall?
    72% aller als überlebend klassifizierten haben wirklich überlebt
#### Was passiert wenn du random_state in train_test_split auf einen anderen Wert setzt? Ändert sich die Accuracy? Warum?
    Die verteiltung der datensätze für train und test datensätze verändert sich und somit auch die antrainierten fehler

# Teil 4:
True Positives (TP) = korrekt vorhergesagte Überlebende
False Positives (FP) = falsch vorhergesagte Überlebende
False Negatives (FN) = falsch vorhergesagte Tote
True Negatives (TN) = korrekt vorhergesagte Tote

#### False Negative ist am häufigsten
#### False Negatives (Lebende person wird als Tot klassifiziert)
    Da dann zb mögliche Rettungsaktionen für diese Person nicht duchgeführt werden würden

# Teil 5:
Schau dir die Eingabewerte und die berechneten Wahrscheinlichkeiten für die ersten 5 Datensätze an.
```
Eingabewerte für die ersten 5 Datensätze + berechnete Wahrscheinlichkeiten für Überleben (Klasse 1)
     pclass    sex   age   fare
149       2   True  42.0  13.00
407       2   True   3.0  18.75
53        2  False  29.0  26.00
369       1  False  24.0  69.30
818       3   True  43.0   6.45
Wahrscheinlichkeiten
[0.16915512 0.50520894 0.81012496 0.94742713 0.04799403]
```
#### Ab welcher Wahrscheinlichkeit wird „überlebt (1)“ vorhergesagt? > Kann man diesen Schwellenwert verändern? Finde heraus, wie man den Schwellenwert verändern kann, um z.B. „überlebt (1)“ schon bei einer Wahrscheinlichkeit von 0.3 vorherzusagen.
ab 0.5 wird Überlebt zurückgegeben \
Man kann ihn nicht wirklich ändern, sondern nur die wahrscheinlichkeit (mittels predict_proba) berechnen und dann händisch ab 0.3 True zurückgeben
#### Finde einen Datenpunkt mit Wahrscheinlichkeit von nahe 0.5. Was sind die Eingabewerte für diesen Datenpunkt?
Datenpunkt:
```
     pclass    sex   age   fare       prob
407       2   True   3.0  18.75 0.50520894
```
#### Welcher Datenpunkt hat die höchste Wahrscheinlichkeit? Was sind die Unterschiede zu einem Datenpunkt mit niedriger Wahrscheinlichkeit?
```
     pclass    sex    age    fare      prob
297       1   False   2.0  151.55  0.971745
851       3   True   74.0   7.775  0.013535
```
Junge weibliche Babies in Klasse 1 sind viel besser aufgestellt als alte Männer in Klasse 3

# Teil 6: Modell verstehen

```
Features: ['pclass', 'sex', 'age', 'fare']
Koeffizienten: [-1.3758619356461192, -2.5419018280907912, -0.0418366168437851, -0.003332965789766604]
```

#### Welche Variable hat den größten Einfluss?
    Das Geschlecht
#### Welche Auswirkung hat das Vorzeichen des Koeffizienten auf die Vorhersage bezügich der Überlebenschancen? Konkret: Was bedeutet ein negativer oder positiver Koeffizient für die Variable age?
    mit negativem Vorzeichen bedeutet das, dass ein höherer Wert die Überlebenschancen mindert
####  Interpretiere konkret für alle Variablen die Auswirkungen auf die Überlebenschancen. Zum Beispiel: „Wenn sich diese Variable erhöht, dann …“
    Wenn das Geschlecht Männlich (=1) statt Weiblich (=0) ist, sinkt die Chance zu Überleben stark
    Wenn die Passagierklasse steigt (von 1 auf 3) sinkt die Chance ebenso, aber nicht so stark
    Ältere Personen haben eine niedriger Überlebenschance
    Wieviel sie fürs Ticket bezahlt haben, hat fast keinen Einfluss

#### Welche Gruppe hat laut Modell bessere Überlebenschancen:
    Deutlich Frauen

# Teil 7: ROC & AUC

#### Wie groß ist euer AUC-Wert?
    0.82
#### Was bedeutet:
    AUC nahe 1?: Das Modell teilt fast mit 100%iger Wahrscheinlichkeit die richtige Klasse zu
    AUC nahe 0.5?: Das Modell ist nur leicht besser als der Zufall

#### Ist das Modell gut? Begründe!
    Ja, da wir mit 80% den richtigen Ausgang vorhersagen

# Teil 8: Reflexion

#### Welche Variable überrascht dich im Modell? (Hinweis: _Betrachte die Koeffizienten und überlege welche Variable den größten Einfluss hat…)
    Dass das Geschlecht mehr Einfluss als die Klasse hat. Ich hätte erwartet, dass Klasse 1 deutlich bessere Chancen als Klasse 2 hätte

#### Was könnte man verbessern?
    mehr Daten? -> eventuell gibt es nicht mehr Datensätze
    andere Features? -> ist crew/nicht
    ein Feature entfernen? -> den Ticketpreis

# Teil 9. Modell anpassen und Auswirkungen beobachten
test_size=0.3 \
Was passiert mit der Accuracy? Warum? \
Sie sinkt in beiden Fällen um 1% \
Wenn die test_size steigt hat das Modell weniger Daten um zu lernen
#### Füge ein neues Feature hinzu:
    df['age_squared'] = df['age']**2

Verbessert sich das Modell?
```
Nein: die Accuracy sinkt um 1%
```

#### Setze max_iter auf einen höheren Wert und beobachte die Auswirkungen: Verwende unterschiedliche Werte, z.B. 500, 1000, 2000:
Ändert sich etwas?
```
Nein: es bleibt alles fast so wie es war, nur das Trainieren dauert jetzt länger
```