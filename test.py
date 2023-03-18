import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


casualty_data = pd.read_csv('casualty_train.csv')
casualty_data = casualty_data.loc[~(casualty_data==-1).any(axis=1)]
#casualty_data = casualty_data.loc[casualty_data.casualty_class == 1]

casualty_X = casualty_data.drop(["casualty_severity", "accident_reference", "lsoa_of_casualty", "sex_of_casualty", "vehicle_reference", "casualty_reference"], axis=1)

# pedestrian; casualty_type == 0
#casualty_X = casualty_data[["age_of_casualty", "pedestrian_location", "pedestrian_movement", "casualty_imd_decile"]]

# driver; casualty_class == 1
#casualty_X = casualty_data[["age_of_casualty", "casualty_type", "casualty_home_area_type", "sex_of_casualty", "casualty_imd_decile"]]

# passenger; casualty_class == 2
#casualty_X = casualty_data[["age_of_casualty", "casualty_type", "casualty_home_area_type", "sex_of_casualty", "casualty_imd_decile", "bus_or_coach_passenger"]]

casualty_Y = casualty_data[["casualty_severity"]].casualty_severity.values.tolist()

"""rus = SMOTE()
casualty_X, casualty_Y = rus.fit_resample(casualty_X, casualty_Y)"""

"""rus = RandomOverSampler(sampling_strategy='auto')
casualty_X, casualty_Y = rus.fit_resample(casualty_X, casualty_Y)"""

rus = RandomUnderSampler(random_state=42, replacement=True)
casualty_X, casualty_Y = rus.fit_resample(casualty_X, casualty_Y)


print(casualty_X)



x_train, x_test, y_train, y_test = train_test_split(casualty_X, casualty_Y, test_size=0.1, random_state=0)





logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(x_train, y_train)

result = logisticRegr.predict(x_test)

cm = confusion_matrix(y_test, result)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logisticRegr.classes_)

display.plot()

print(classification_report(y_test, result))
#plot_confusion_matrix(logisticRegr, x_test, y_test, cmap=plt.cm.Blues)
plt.show()


"""print(pd.DataFrame({
    'Predictions': result,
    'Actual Results': y_test,
    'INCORRECT?': abs(result - y_test)
}))"""
