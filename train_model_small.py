import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("KaggleV2-May-2016.csv")

# Feature engineering
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['DaysWaiting'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df[df['DaysWaiting'] >= 0]

# Drop irrelevant columns
df.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

# Split features and label
X = df.drop('No-show', axis=1)
y = df['No-show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train smaller model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save smaller model
joblib.dump(model, "model_small.pkl")

print("✅ Smaller model saved as model_small.pkl")


