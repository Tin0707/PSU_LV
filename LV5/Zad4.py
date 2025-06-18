import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv('C:/Users/brnar/Downloads/occupancy_processed.csv')

X = df[['S3_Temp', 'S5_CO2']].to_numpy()
y = df['Room_Occupancy_Count'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Točnost modela:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Slobodna', 'Zauzeta'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrica zabune')
plt.show()
